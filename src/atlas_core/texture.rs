use crate::atlas_core::mesh::Texture;
use crate::atlas_core::System;
use png::ColorType;
use std::io::prelude::*;
use std::{fs::File, io::Cursor, sync::Arc};
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::sampler::Sampler;
use vulkano::sampler::{Filter, SamplerAddressMode, SamplerCreateInfo};
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

    let (width, height) = (info.width, info.height);
    let array_layers = 1;

    let color_type = info.color_type;
    let has_alpha = color_type == ColorType::Rgba;
    let channel_count = if has_alpha { 4 } else { 3 };

    let mut image_data = Vec::new();
    image_data.resize((info.width * info.height * channel_count) as usize, 0);
    reader.next_frame(&mut image_data).unwrap();

    let image_data_alpha: Vec<u8> = if has_alpha {
        image_data
    } else {
        image_data
            .chunks(3)
            .map(|x| [x[0], x[1], x[2], 255])
            .flatten()
            .collect()
    };

    let (image, future) = ImmutableImage::from_iter(
        image_data_alpha,
        ImageDimensions::Dim2d {
            width,
            height,
            array_layers,
        },
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

pub fn get_descriptor_set(
    system: &System,
    layout: &Arc<DescriptorSetLayout>,
    texture: Texture,
) -> Arc<PersistentDescriptorSet> {
    let image = texture.image;

    let sampler = Sampler::new(
        system.device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            ..Default::default()
        },
    )
    .unwrap();

    PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view_sampler(
            0,
            image.clone(),
            sampler.clone(),
        )],
    )
    .unwrap()
}
