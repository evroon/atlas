use crate::atlas_core::texture::load_png;
use crate::atlas_core::System;
use crate::CpuAccessibleBuffer;
use bytemuck::{Pod, Zeroable};
use russimp::scene::{PostProcess, Scene};
use russimp::texture::DataContent;
use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::image::view::ImageView;
use vulkano::image::ImmutableImage;
use vulkano::impl_vertex;
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
    pub coordinate: [f32; 2],
}

impl_vertex!(TexCoord, coordinate);

pub struct MeshBuffer {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub normal_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub tex_coord_buffer: Option<Arc<CpuAccessibleBuffer<[TexCoord]>>>,
    pub material_index: u32,
}

pub struct Texture {
    pub image: Arc<ImageView<ImmutableImage>>,
    pub future: CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>,
}

pub struct Material {
    pub textures: Vec<Texture>,
}

pub struct Mesh {
    pub mesh_buffers: Vec<MeshBuffer>,
    pub materials: Vec<Material>,
}

pub fn load_gltf(system: &System) -> Mesh {
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

    let mut materials: Vec<Material> = vec![];
    let mut mesh_buffers: Vec<MeshBuffer> = vec![];

    for assimp_material in &scene.materials {
        let mut textures: Vec<Texture> = vec![];

        for assimp_texture in &assimp_material.textures {
            let texture_info = &assimp_texture.1[0];
            assert_eq!(
                texture_info.ach_format_hint, "png",
                "Encompassed texture data should be in png format"
            );

            let texture = match texture_info.data.as_ref().expect("Unexpected texture data") {
                DataContent::Texel(_) => panic!("Loading textures by texels is not yet supported"),
                DataContent::Bytes(bytes) => load_png(&system.queue, bytes),
            };
            textures.push(texture);
        }
        materials.push(Material { textures });
    }

    for mesh in &scene.meshes {
        let assimp_vertices = &mesh.vertices;
        let assimp_normals = &mesh.normals;
        let assimp_faces = &mesh.faces;
        let assimp_tex_coords = &mesh.texture_coords;

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

        let mut tex_coord_buffer: Option<Arc<CpuAccessibleBuffer<[TexCoord]>>> = None;

        if assimp_tex_coords.into_iter().all(|x| (x).is_some()) {
            let tex_coords: Vec<TexCoord> = assimp_tex_coords
                .iter()
                .map(|tc| tc.as_ref().unwrap()[0])
                .map(|tc| TexCoord {
                    coordinate: [tc.x, tc.y],
                })
                .collect();
            tex_coord_buffer = Some(
                CpuAccessibleBuffer::from_iter(
                    system.device.clone(),
                    BufferUsage::all(),
                    false,
                    tex_coords,
                )
                .unwrap(),
            );
        }

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

        mesh_buffers.push(MeshBuffer {
            vertex_buffer,
            normal_buffer,
            index_buffer,
            tex_coord_buffer,
            material_index: mesh.material_index,
        });
    }

    Mesh {
        mesh_buffers,
        materials,
    }
}
