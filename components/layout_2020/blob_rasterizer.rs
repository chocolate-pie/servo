use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::mem::ManuallyDrop;

use euclid::point2;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use tiny_skia::{Mask, Paint, Pixmap, Rect, FillRule, Transform, PathBuilder};
use webrender_api::units::{BlobDirtyRect, BlobToDeviceTranslation, DeviceIntRect, LayoutPoint, LayoutRect};
use webrender_api::{
    AsyncBlobImageRasterizer, BlobImageData, BlobImageHandler, BlobImageKey, BlobImageParams,
    BlobImageRequest, BlobImageResult, DirtyRect, ImageFormat, RasterizedBlobImage, TileSize,
};

#[derive(Debug, Clone)]
pub enum BlobImageCommandKind {
    FillRect,
    DrawPolygon(ManuallyDrop<Vec<webrender_api::units::LayoutPoint>>)
}

#[derive(Clone, Debug)]
pub struct BlobImageCommand {
    pub kind: BlobImageCommandKind,
    pub bounds: LayoutRect,
}

#[allow(unsafe_code)]
fn convert_to_bytes<T>(x: &T) -> &[u8] {
    let pointer = x as *const _ as *const u8;
    unsafe { std::slice::from_raw_parts(pointer, std::mem::size_of::<T>()) }
}

#[allow(unsafe_code)]
fn convert_from_bytes<T>(x: &[u8]) -> T {
    assert!(std::mem::size_of::<T>() <= x.len());
    unsafe { std::ptr::read_unaligned(x.as_ptr() as *const T) }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct BlobDataHeader {
    length: usize,
}

#[derive(Debug)]
pub struct BlobCommand {
    data: Arc<BlobImageData>,
    visible_rect: DeviceIntRect,
    #[allow(dead_code)]
    tile_size: TileSize,
}

pub struct BlobData(Vec<u8>);
pub struct BlobDataIterator<'a> {
    current_pos: usize,
    data: &'a [u8],
}

impl BlobData {
    pub fn new() -> BlobData {
        let mut buffer = BlobData(Vec::new());
        buffer.0.resize(std::mem::size_of::<BlobDataHeader>(), 0);
        buffer.write_header(BlobDataHeader::default());
        buffer
    }

    pub fn new_with_capacity(capacity: usize) -> BlobData {
        let capacity = capacity * std::mem::size_of::<BlobImageCommand>();
        let capacity = std::mem::size_of::<BlobDataHeader>() + capacity;
        let mut buffer = BlobData(Vec::with_capacity(capacity));
        buffer.0.resize(std::mem::size_of::<BlobDataHeader>(), 0);
        buffer.write_header(BlobDataHeader::default());
        buffer
    }

    pub fn new_entry(&mut self, data: BlobImageCommand) -> usize {
        let header = convert_from_bytes::<BlobDataHeader>(&self.0);
        let data = convert_to_bytes::<BlobImageCommand>(&data);
        let new_header = BlobDataHeader {
            length: header.length + 1,
        };
        self.0.extend_from_slice(&data);
        self.write_header(new_header);
        header.length
    }

    pub fn update_entry(&mut self, index: usize, data: BlobImageCommand) {
        let data = convert_to_bytes::<BlobImageCommand>(&data);
        let size = std::mem::size_of::<BlobImageCommand>();
        let offset = std::mem::size_of::<BlobDataHeader>() + (index * size);
        self.0[offset..offset + size].copy_from_slice(&data);
    }

    fn write_header(&mut self, header: BlobDataHeader) {
        let header = convert_to_bytes(&header);
        self.0[..header.len()].copy_from_slice(header);
    }

    #[inline(always)]
    pub fn take(self) -> Vec<u8> {
        self.0
    }
}

impl<'a> BlobDataIterator<'a> {
    pub fn from_raw(buffer: &'a [u8]) -> Self {
        Self {
            current_pos: 0,
            data: buffer,
        }
    }
}

impl Iterator for BlobDataIterator<'_> {
    type Item = BlobImageCommand;

    fn next(&mut self) -> Option<Self::Item> {
        let header = convert_from_bytes::<BlobDataHeader>(self.data);
        if self.current_pos < header.length {
            let offset = std::mem::size_of::<BlobImageCommand>() * self.current_pos;
            let offset = std::mem::size_of::<BlobDataHeader>() + offset;
            let command = convert_from_bytes::<BlobImageCommand>(&self.data[offset..]);
            self.current_pos += 1;
            Some(command)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct ServoBlobImageHandler {
    workers: Arc<ThreadPool>,
    enable_multithreading: bool,
    blob_commands: Arc<Mutex<HashMap<BlobImageKey, BlobCommand>>>,
}

#[derive(Debug)]
pub struct ServoBlobRasterizer {
    workers: Arc<ThreadPool>,
    enable_multithreading: bool,
    blob_commands: Arc<Mutex<HashMap<BlobImageKey, BlobCommand>>>,
}

impl ServoBlobImageHandler {
    pub fn new() -> ServoBlobImageHandler {
        let thread_count = std::thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(4).unwrap())
            .get();
        let workers = ThreadPoolBuilder::new()
            .thread_name(|i| format!("BlobImageHandler#{i}"))
            .num_threads(thread_count)
            .build()
            .unwrap();
        Self {
            workers: Arc::new(workers),
            enable_multithreading: true,
            blob_commands: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl BlobImageHandler for ServoBlobImageHandler {
    fn create_similar(&self) -> Box<dyn BlobImageHandler> {
        Box::new(Self {
            workers: self.workers.clone(),
            enable_multithreading: self.enable_multithreading,
            blob_commands: self.blob_commands.clone(),
        })
    }

    fn create_blob_rasterizer(&mut self) -> Box<dyn AsyncBlobImageRasterizer> {
        Box::new(ServoBlobRasterizer {
            workers: self.workers.clone(),
            enable_multithreading: self.enable_multithreading,
            blob_commands: self.blob_commands.clone(),
        })
    }

    fn add(
        &mut self,
        key: BlobImageKey,
        data: Arc<BlobImageData>,
        visible_rect: &DeviceIntRect,
        tile_size: TileSize,
    ) {
        self.blob_commands.lock().unwrap().insert(
            key,
            BlobCommand {
                data: data.clone(),
                visible_rect: *visible_rect,
                tile_size,
            },
        );
    }

    fn update(
        &mut self,
        key: BlobImageKey,
        data: Arc<BlobImageData>,
        visible_rect: &DeviceIntRect,
        dirty_rect: &BlobDirtyRect,
    ) {
        if let Some(command) = self.blob_commands.lock().unwrap().get_mut(&key) {
            let dirty_rect = match dirty_rect {
                DirtyRect::All => DeviceIntRect {
                    min: point2(i32::MIN, i32::MIN),
                    max: point2(i32::MAX, i32::MAX),
                },
                DirtyRect::Partial(d) => d.cast_unit(),
            };
            let mut new_blob_data = BlobData::new();
            let new_blob_data_iter = BlobDataIterator::from_raw(&data);
            let preserved_rect = command.visible_rect.intersection_unchecked(visible_rect);
            for blob_data in new_blob_data_iter {
                if dirty_rect.contains_box(&preserved_rect) {
                    new_blob_data.new_entry(blob_data);
                }
            }
            command.data = Arc::new(new_blob_data.0);
            command.visible_rect = *visible_rect;
        }
    }

    fn delete(&mut self, key: BlobImageKey) {
        self.blob_commands.lock().unwrap().remove(&key);
    }

    fn enable_multithreading(&mut self, enable: bool) {
        self.enable_multithreading = enable;
    }

    fn prepare_resources(
        &mut self,
        _services: &dyn webrender_api::BlobImageResources,
        _requests: &[webrender_api::BlobImageParams],
    ) {
    }
    fn delete_font(&mut self, _key: webrender_api::FontKey) {}
    fn clear_namespace(&mut self, _namespace: webrender_api::IdNamespace) {}
    fn delete_font_instance(&mut self, _key: webrender_api::FontInstanceKey) {}
}

impl ServoBlobRasterizer {
    fn process_blob(&self, pixmap: &mut Pixmap, command: BlobImageCommand) {
        match command.kind {
            BlobImageCommandKind::FillRect => pixmap.fill_rect(
                self.to_tiny_skia_rect(command.bounds),
                &Paint::default(),
                Transform::identity(),
                None,
            ),
            BlobImageCommandKind::DrawPolygon(coordinates) => {
                let mut coordinates = <Vec<LayoutPoint> as Clone>::clone(&coordinates.clone()).into_iter();
                let mut path_builder = PathBuilder::new();
                if let Some(coordinate) = coordinates.next() {
                    path_builder.move_to(coordinate.x, coordinate.y);
                }
                for coordinate in coordinates {
                    path_builder.line_to(coordinate.x, coordinate.y);
                }
                path_builder.close();
                let path = path_builder.finish().unwrap();
                pixmap.fill_path(
                    &path,
                    &Paint::default(),
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                )
            }
        }
    }

    fn to_tiny_skia_rect(&self, bounds: LayoutRect) -> Rect {
        Rect::from_xywh(bounds.min.x, bounds.min.y, bounds.width(), bounds.height()).unwrap()
    }

    fn rasterize_blob(&self, request: &BlobImageParams) -> (BlobImageRequest, BlobImageResult) {
        match request.descriptor.format {
            ImageFormat::RGBA8 => {},
            _ => unimplemented!(),
        }
        let rect = request.descriptor.rect;
        let mut pixmap = Pixmap::new(rect.width() as u32, rect.height() as u32).unwrap();
        let command = &self.blob_commands.lock().unwrap()[&request.request.key];
        let blob_data = BlobDataIterator::from_raw(&command.data);
        let dirty_rect = match request.dirty_rect {
            DirtyRect::Partial(rect) => Some(rect),
            DirtyRect::All => None,
        };
        for mut command in blob_data {
            if let Some(ref dirty_rect) = dirty_rect {
                command.bounds = command.bounds.intersection_unchecked(&dirty_rect.cast());
            }
            command.bounds = LayoutRect::from_size(command.bounds.size());
            println!("COMMAND: {command:?}");
            self.process_blob(&mut pixmap, command);
        }

        #[cfg(debug_assertions)]
        {
            if let Ok(out) = std::env::var("SERVO_BLOB_OUTPUT_DIR") {
                use std::path::Path;
                use std::sync::atomic::{AtomicU32, Ordering};
                static RASTERIZED_BLOB_COUNT: AtomicU32 = AtomicU32::new(0);
                let filename = format!(
                    "rasterized-blob-{}.png",
                    RASTERIZED_BLOB_COUNT.load(Ordering::SeqCst)
                );
                let _ = pixmap.save_png(Path::new(&out).join(filename));
                RASTERIZED_BLOB_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        let dirty_rect = request.dirty_rect.to_subrect_of(&rect);
        let tx: BlobToDeviceTranslation = (-rect.min.to_vector()).into();
        let rasterized_rect = tx.transform_box(&dirty_rect);
        (
            request.request,
            Ok(RasterizedBlobImage {
                rasterized_rect,
                data: Arc::new(pixmap.take()),
            }),
        )
    }
}

impl AsyncBlobImageRasterizer for ServoBlobRasterizer {
    fn rasterize(
        &mut self,
        requests: &[BlobImageParams],
        _low_priority: bool,
    ) -> Vec<(BlobImageRequest, BlobImageResult)> {
        if self.enable_multithreading {
            let thread_task = || {
                requests
                    .into_par_iter()
                    .map(|r| self.rasterize_blob(r))
                    .collect()
            };
            self.workers.install(thread_task)
        } else {
            requests
                .into_iter()
                .map(|r| self.rasterize_blob(r))
                .collect()
        }
    }
}
