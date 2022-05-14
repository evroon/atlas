use crate::atlas_core::mesh::Texture;
use std::io::prelude::*;
use std::{fs::File, io::Cursor, sync::Arc};
use vulkano::{
    device::Queue,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
};

#[allow(dead_code)]
pub fn load_png(queue: &Arc<Queue>, data: &Vec<u8>) -> Texture {
    let cursor = Cursor::new(data);
    let decoder = png::Decoder::new(cursor);
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info();
    let dimensions = ImageDimensions::Dim2d {
        width: info.width,
        height: info.height,
        array_layers: 1,
    };
    let mut image_data = Vec::new();
    image_data.resize((info.width * info.height * 4) as usize, 0);
    reader.next_frame(&mut image_data).unwrap();

    let (image, future) = ImmutableImage::from_iter(
        image_data,
        dimensions,
        MipmapsCount::One,
        Format::R8G8B8A8_SRGB,
        queue.clone(),
    )
    .unwrap();

    Texture {
        image: ImageView::new_default(image).unwrap(),
        future,
    }
}

#[allow(dead_code)]
pub fn load_png_file(queue: &Arc<Queue>, path: &str) -> Texture {
    let mut f = File::open(path).expect("Could not open file");
    let mut png_bytes = Vec::new();

    f.read_to_end(&mut png_bytes)
        .expect("Could not read png file");

    load_png(queue, &png_bytes)
}
