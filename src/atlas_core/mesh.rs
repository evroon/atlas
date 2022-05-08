use crate::atlas_core::System;
use crate::CpuAccessibleBuffer;
use bytemuck::{Pod, Zeroable};
use russimp::scene::{PostProcess, Scene};
use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::impl_vertex;

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
}

pub fn load_gltf(system: &System) -> MeshBuffer {
    let scene = Scene::from_file(
        "assets/models/monkey.glb",
        vec![
            PostProcess::CalculateTangentSpace,
            PostProcess::Triangulate,
            PostProcess::JoinIdenticalVertices,
            PostProcess::SortByPrimitiveType,
        ],
    )
    .expect("Could not load model");

    let assimp_vertices = &scene.meshes[0].vertices;
    let assimp_normals = &scene.meshes[0].normals;
    let assimp_faces = &scene.meshes[0].faces;
    let assimp_tex_coords = &scene.meshes[0].texture_coords;

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

    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(system.device.clone(), BufferUsage::all(), false, vertices)
            .unwrap();
    let normal_buffer =
        CpuAccessibleBuffer::from_iter(system.device.clone(), BufferUsage::all(), false, normals)
            .unwrap();
    let index_buffer =
        CpuAccessibleBuffer::from_iter(system.device.clone(), BufferUsage::all(), false, indices)
            .unwrap();

    MeshBuffer {
        vertex_buffer,
        normal_buffer,
        index_buffer,
        tex_coord_buffer,
    }
}
