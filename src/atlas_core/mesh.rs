use crate::atlas_core::texture::get_descriptor_set;
use crate::atlas_core::texture::load_png;
use crate::atlas_core::texture::load_png_file;
use crate::atlas_core::System;
use crate::CpuAccessibleBuffer;
use crate::PersistentDescriptorSet;
use bytemuck::{Pod, Zeroable};
use russimp::scene::{PostProcess, Scene};
use russimp::texture::DataContent;
use russimp::texture::TextureType;
use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::image::view::ImageView;
use vulkano::image::ImmutableImage;
use vulkano::impl_vertex;
use vulkano::sync::now;
use vulkano::sync::NowFuture;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}

impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    pub normal: [f32; 3],
}

impl_vertex!(Normal, normal);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct TexCoord {
    pub tex_coord: [f32; 2],
}

impl_vertex!(TexCoord, tex_coord);

pub struct MeshBuffer {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub normal_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub tex_coord_buffer: Arc<CpuAccessibleBuffer<[TexCoord]>>,
    pub material: Material,
}

pub struct Texture {
    pub image: Arc<ImageView<ImmutableImage>>,
    pub future: CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>,
}

pub struct Material {
    pub textures: Vec<Texture>,
    pub uniform_set: Option<Arc<PersistentDescriptorSet>>,
}

pub struct Mesh {
    pub mesh_buffers: Vec<MeshBuffer>,
    pub materials: Vec<Material>,
}

pub fn load_material(
    system: &System,
    layout: &Arc<DescriptorSetLayout>,
    assimp_material: &russimp::material::Material,
) -> Material {
    let mut textures: Vec<Texture> = vec![];
    let default_texture = load_png_file(&system.queue, "assets/models/sponza/textures/test.png");
    let base_textures = assimp_material.textures.get(&TextureType::BaseColor);

    let result_tex = if base_textures.is_some() {
        let assimp_texture = &base_textures.unwrap().first().unwrap();
        
        assert_eq!(
            assimp_texture.ach_format_hint, "png",
            "Encompassed texture data should be in png format"
        );

        let texture = match assimp_texture
            .data
            .as_ref()
            .expect("Unexpected texture data")
        {
            DataContent::Texel(_) => panic!("Loading textures by texels is not yet supported"),
            DataContent::Bytes(bytes) => load_png(&system.queue, bytes),
        };

        Some(texture)
    } else {
        None
    };

    let uniform_set = match result_tex {
        None => get_descriptor_set(system, layout, default_texture),
        Some(x) => get_descriptor_set(system, layout, x),
    };

    Material {
        textures,
        uniform_set: Some(uniform_set),
    }
}

pub fn load_gltf(system: &System, layout: &Arc<DescriptorSetLayout>) -> Mesh {
    let scene = Scene::from_file(
        "assets/models/sponza/sponza.glb",
        vec![
            PostProcess::CalculateTangentSpace,
            PostProcess::Triangulate,
            PostProcess::JoinIdenticalVertices,
            PostProcess::SortByPrimitiveType,
        ],
    )
    .expect("Could not load model");

    let materials: Vec<Material> = vec![];
    let mut mesh_buffers: Vec<MeshBuffer> = vec![];
    let default_textures: Vec<Texture> = vec![load_png_file(
        &system.queue,
        "assets/models/sponza/textures/test.png",
    )];

    for mesh in &scene.meshes {
        let assimp_vertices = &mesh.vertices;
        let assimp_normals = &mesh.normals;
        let assimp_faces = &mesh.faces;
        let assimp_tex_coords = &mesh.texture_coords;
        let material = load_material(
            system,
            layout,
            scene.materials.get(mesh.material_index as usize).unwrap(),
            &default_textures,
        );

        let vertices: Vec<Vertex> = assimp_vertices
            .iter()
            .map(|v| Vertex {
                position: [v.x, v.y, v.z],
            })
            .collect();
        let normals: Vec<Normal> = assimp_normals
            .iter()
            .map(|v| Normal {
                normal: [v.x, v.y, v.z],
            })
            .collect();
        let indices: Vec<u32> = assimp_faces
            .iter()
            .map(|f| [f.0[0], f.0[1], f.0[2]])
            .flatten()
            .collect();

        if assimp_tex_coords[0].is_none() {
            panic!("Could not find texture coordinates");
        }

        let tex_coords: Vec<TexCoord> = assimp_tex_coords[0]
            .as_ref()
            .unwrap()
            .into_iter()
            .map(|tc| TexCoord {
                tex_coord: [tc.x, 1.0 - tc.y],
            })
            .collect();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            system.device.clone(),
            BufferUsage::all(),
            false,
            vertices,
        )
        .unwrap();
        let normal_buffer = CpuAccessibleBuffer::from_iter(
            system.device.clone(),
            BufferUsage::all(),
            false,
            normals,
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            system.device.clone(),
            BufferUsage::all(),
            false,
            indices,
        )
        .unwrap();
        let tex_coord_buffer = CpuAccessibleBuffer::from_iter(
            system.device.clone(),
            BufferUsage::all(),
            false,
            tex_coords,
        )
        .unwrap();

        mesh_buffers.push(MeshBuffer {
            vertex_buffer,
            normal_buffer,
            index_buffer,
            tex_coord_buffer,
            material,
        });
    }

    Mesh {
        mesh_buffers,
        materials,
    }
}
