use std::cmp::Ordering;
use std::sync::Arc;
use std::mem::ManuallyDrop;

use style::values::computed::basic_shape::{BasicShape, ClipPath, FillRule};
use style::values::computed::length::Length;
use style::values::computed::length_percentage::{LengthPercentage, NonNegativeLengthPercentage};
use style::values::computed::position::Position;
use style::values::generics::basic_shape::{
    GenericPolygon, GenericShapeRadius, ShapeBox, ShapeGeometryBox,
};
use style::values::generics::position::GenericPositionOrAuto;
use webrender_api::units::{LayoutPoint, LayoutRect, LayoutSize};
use webrender_api::{
    BlobImageKey, ClipChainId, FillRule as WrFillRule, ImageDescriptor, ImageDescriptorFlags,
    ImageFormat, ImageMask, POLYGON_CLIP_VERTEX_MAX,
};
use webrender_traits::display_list::ScrollTreeNodeId;
use webrender_traits::ImageUpdate;

use super::{compute_marginbox_radius, normalize_radii};
use crate::blob_rasterizer::{BlobData, BlobImageCommand, BlobImageCommandKind};

pub(super) fn build(
    clip_path: ClipPath,
    display_list: &mut super::DisplayList,
    parent_scroll_node_id: ScrollTreeNodeId,
    parent_clip_chain_id: ClipChainId,
    fragment_builder: super::BuilderForBoxFragment,
) -> Option<ClipChainId> {
    let geometry_box = match clip_path {
        ClipPath::Shape(_, ShapeGeometryBox::ShapeBox(shape_box)) => shape_box,
        ClipPath::Shape(_, ShapeGeometryBox::ElementDependent) => ShapeBox::BorderBox,
        ClipPath::Box(ShapeGeometryBox::ShapeBox(shape_box)) => shape_box,
        ClipPath::Box(ShapeGeometryBox::ElementDependent) => ShapeBox::BorderBox,
        _ => return None,
    };
    if let ClipPath::Shape(shape, _) = clip_path {
        let layout_rect = match geometry_box {
            ShapeBox::BorderBox => fragment_builder.border_rect,
            ShapeBox::ContentBox => *fragment_builder.content_rect(),
            ShapeBox::PaddingBox => *fragment_builder.padding_rect(),
            ShapeBox::MarginBox => *fragment_builder.margin_rect(),
        };
        match *shape {
            BasicShape::Polygon(polygon) => build_polygon(
                polygon,
                layout_rect,
                parent_scroll_node_id,
                parent_clip_chain_id,
                display_list,
            ),
            BasicShape::Circle(_) | BasicShape::Ellipse(_) | BasicShape::Rect(_) => {
                build_simple_shape(
                    *shape,
                    layout_rect,
                    parent_scroll_node_id,
                    parent_clip_chain_id,
                    display_list,
                )
            },
            BasicShape::PathOrShape(_) => None,
        }
    } else {
        let layout_rect = match geometry_box {
            ShapeBox::BorderBox => fragment_builder.border_rect,
            ShapeBox::ContentBox => *fragment_builder.content_rect(),
            ShapeBox::PaddingBox => *fragment_builder.padding_rect(),
            ShapeBox::MarginBox => *fragment_builder.margin_rect(),
        };
        create_rect_clip_chain(
            match geometry_box {
                ShapeBox::MarginBox => compute_marginbox_radius(
                    fragment_builder.border_radius,
                    layout_rect.size(),
                    fragment_builder.fragment,
                ),
                _ => fragment_builder.border_radius,
            },
            layout_rect,
            parent_scroll_node_id,
            parent_clip_chain_id,
            display_list,
        )
    }
}

fn build_polygon(
    polygon: GenericPolygon<LengthPercentage>,
    layout_rect: LayoutRect,
    parent_scroll_node_id: ScrollTreeNodeId,
    parent_clip_chain_id: ClipChainId,
    display_list: &mut super::DisplayList,
) -> Option<ClipChainId> {
    if polygon.coordinates.len() > POLYGON_CLIP_VERTEX_MAX {
        return None;
    }
    let webrender_api_sender = &display_list.webrender_api_sender;
    let mut bounds = None;
    let mut points = Vec::with_capacity(polygon.coordinates.len());
    let mut coordinates = Vec::with_capacity(polygon.coordinates.len());
    for coordinate in polygon.coordinates {
        let x = coordinate.0.resolve(Length::new(layout_rect.width()));
        let y = coordinate.1.resolve(Length::new(layout_rect.height()));
        let point = LayoutPoint::new(x.px(), y.px());
        let bounds = bounds.get_or_insert(LayoutRect::new(point, point));
        *bounds = LayoutRect::new(bounds.min.min(point), bounds.max.max(point));
        let coord = LayoutPoint::new(x.px() + layout_rect.min.x, y.px() + layout_rect.min.y);
        points.push(point);
        coordinates.push(coord);
    }
    let bounds = bounds?;
    let mut updates = Vec::with_capacity(1);
    let mut blob_data = BlobData::new_with_capacity(1);
    for point in &mut points {
        *point = *point - bounds.min.to_vector();
    }
    let command = BlobImageCommand {
        kind: BlobImageCommandKind::DrawPolygon(ManuallyDrop::new(points)),
        bounds: layout_rect,
    };
    let descriptor = ImageDescriptor::new(
        bounds.width() as i32,
        bounds.height() as i32,
        ImageFormat::RGBA8,
        ImageDescriptorFlags::IS_OPAQUE,
    );
    let image_mask = ImageMask {
        image: webrender_api_sender.generate_image_key()?,
        rect: bounds.translate(layout_rect.min.to_vector()),
    };
    let fill = match polygon.fill {
        FillRule::Evenodd => WrFillRule::Evenodd,
        FillRule::Nonzero => WrFillRule::Nonzero,
    };
    let blob_key = BlobImageKey(image_mask.image);
    let spatial_id = parent_scroll_node_id.spatial_id;
    blob_data.new_entry(command);
    updates.push(ImageUpdate::AddBlobImage(
        blob_key,
        descriptor,
        Arc::new(blob_data.take()),
    ));
    webrender_api_sender.update_images(updates);
    let new_clip_id =
        display_list
            .wr
            .define_clip_image_mask(spatial_id, image_mask, &coordinates, fill);
    Some(display_list.define_clip_chain(parent_clip_chain_id, [new_clip_id]))
}

fn build_simple_shape(
    shape: BasicShape,
    layout: LayoutRect,
    parent_scroll_node_id: ScrollTreeNodeId,
    parent_clip_chain_id: ClipChainId,
    display_list: &mut super::DisplayList,
) -> Option<ClipChainId> {
    match shape {
        BasicShape::Rect(rect) => {
            let top = rect.rect.0.resolve(Length::new(layout.height()));
            let right = rect.rect.1.resolve(Length::new(layout.width()));
            let bottom = rect.rect.2.resolve(Length::new(layout.height()));
            let left = rect.rect.3.resolve(Length::new(layout.width()));
            let x = layout.min.x + left.px();
            let y = layout.min.y + top.px();
            let width = layout.width() - (left + right).px();
            let height = layout.height() - (top + bottom).px();
            let resolve = |radius: &LengthPercentage, box_size: f32| {
                radius.percentage_relative_to(Length::new(box_size)).px()
            };
            let corner = |corner: &style::values::computed::BorderCornerRadius| {
                LayoutSize::new(
                    resolve(&corner.0.width.0, layout.size().width),
                    resolve(&corner.0.height.0, layout.size().height),
                )
            };
            let mut radii = webrender_api::BorderRadius {
                top_left: corner(&rect.round.top_left),
                top_right: corner(&rect.round.top_right),
                bottom_left: corner(&rect.round.bottom_left),
                bottom_right: corner(&rect.round.bottom_right),
            };
            let origin = LayoutPoint::new(x, y);
            let size = LayoutSize::new(width, height);
            let rect = LayoutRect::from_origin_and_size(origin, size);
            normalize_radii(&layout, &mut radii);
            create_rect_clip_chain(
                radii,
                rect,
                parent_scroll_node_id,
                parent_clip_chain_id,
                display_list,
            )
        },
        BasicShape::Circle(circle) => {
            let center = match circle.position {
                GenericPositionOrAuto::Position(position) => position,
                GenericPositionOrAuto::Auto => Position::center(),
            };
            let anchor_x = center.horizontal.resolve(Length::new(layout.width()));
            let anchor_y = center.vertical.resolve(Length::new(layout.height()));
            let amount = LayoutSize::new(anchor_x.px(), anchor_y.px());
            let center = layout.min.add_size(&amount);
            let horizontal =
                compute_shape_radius(center.x, &circle.radius, layout.min.x, layout.max.x);
            let vertical =
                compute_shape_radius(center.y, &circle.radius, layout.min.y, layout.max.y);
            let radius = match circle.radius {
                GenericShapeRadius::FarthestSide => horizontal.max(vertical),
                _ => horizontal.min(vertical),
            };
            let radius = LayoutSize::new(radius, radius);
            let mut radii = webrender_api::BorderRadius {
                top_left: radius,
                top_right: radius,
                bottom_left: radius,
                bottom_right: radius,
            };
            let start = center.add_size(&-radius);
            let rect = LayoutRect::from_origin_and_size(start, radius * 2.);
            normalize_radii(&layout, &mut radii);
            create_rect_clip_chain(
                radii,
                rect,
                parent_scroll_node_id,
                parent_clip_chain_id,
                display_list,
            )
        },
        BasicShape::Ellipse(ellipse) => {
            let center = match ellipse.position {
                GenericPositionOrAuto::Position(position) => position,
                GenericPositionOrAuto::Auto => Position::center(),
            };
            let anchor_x = center.horizontal.resolve(Length::new(layout.width()));
            let anchor_y = center.vertical.resolve(Length::new(layout.height()));
            let amount = LayoutSize::new(anchor_x.px(), anchor_y.px());
            let center = layout.min.add_size(&amount);
            let width = if let GenericShapeRadius::Length(length) = ellipse.semiaxis_x {
                length.0.resolve(Length::new(layout.width())).px()
            } else {
                compute_shape_radius(center.x, &ellipse.semiaxis_x, layout.min.x, layout.max.x)
            };
            let height = if let GenericShapeRadius::Length(length) = ellipse.semiaxis_y {
                length.0.resolve(Length::new(layout.height())).px()
            } else {
                compute_shape_radius(center.y, &ellipse.semiaxis_y, layout.min.y, layout.max.y)
            };
            let mut radii = webrender_api::BorderRadius {
                top_left: LayoutSize::new(width, height),
                top_right: LayoutSize::new(width, height),
                bottom_left: LayoutSize::new(width, height),
                bottom_right: LayoutSize::new(width, height),
            };
            let size = LayoutSize::new(width, height);
            let start = center.add_size(&-size);
            let rect = LayoutRect::from_origin_and_size(start, size * 2.);
            normalize_radii(&rect, &mut radii);
            create_rect_clip_chain(
                radii,
                rect,
                parent_scroll_node_id,
                parent_clip_chain_id,
                display_list,
            )
        },
        _ => None,
    }
}

fn compute_shape_radius(
    center: f32,
    radius: &GenericShapeRadius<NonNegativeLengthPercentage>,
    layout_min: f32,
    layout_max: f32,
) -> f32 {
    let left = (layout_min - center).abs();
    let right = (layout_max - center).abs();
    match (radius, left.partial_cmp(&right)) {
        (GenericShapeRadius::FarthestSide, Some(Ordering::Greater)) => left,
        (GenericShapeRadius::FarthestSide, _) => right,
        (_, Some(Ordering::Greater)) => right,
        (_, _) => left,
    }
}

fn create_rect_clip_chain(
    radii: webrender_api::BorderRadius,
    rect: LayoutRect,
    parent_scroll_node_id: ScrollTreeNodeId,
    parent_clip_chain_id: ClipChainId,
    display_list: &mut super::DisplayList,
) -> Option<ClipChainId> {
    let new_clip_id = if radii.is_zero() {
        display_list
            .wr
            .define_clip_rect(parent_scroll_node_id.spatial_id, rect)
    } else {
        display_list.wr.define_clip_rounded_rect(
            parent_scroll_node_id.spatial_id,
            webrender_api::ComplexClipRegion {
                rect,
                radii,
                mode: webrender_api::ClipMode::Clip,
            },
        )
    };
    Some(display_list.define_clip_chain(parent_clip_chain_id, [new_clip_id]))
}
