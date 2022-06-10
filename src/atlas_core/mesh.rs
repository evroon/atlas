use crate::atlas_core::texture::get_descriptor_set;
use crate::atlas_core::texture::load_png;
use crate::atlas_core::texture::load_png_file;
use crate::atlas_core::System;
use crate::CpuAccessibleBuffer;
use crate::PersistentDescriptorSet;
use bytemuck::{Pod, Zeroable};
use cgmath::Matrix4;
use russimp::scene::{PostProcess, Scene};
use russimp::texture::DataContent;
use russimp::texture::TextureType;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::TypedBufferAccess;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::image::view::ImageView;
use vulkano::image::ImmutableImage;
use vulkano::impl_vertex;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::NowFuture;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex2D {
    pub position: [f32; 2],
}
impl_vertex!(Vertex2D, position);

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
    pub uniform_set: Option<Arc<PersistentDescriptorSet>>,
}

pub struct Mesh {
    pub mesh_buffers: Vec<MeshBuffer>,
    pub materials: Vec<Material>,
    pub model_matrix: Matrix4<f32>,
}

fn load_default_texture(system: &System) -> Texture {
    load_png_file(
        &system.queue,
        "assets/models/sponza/16011208436118768083.png",
    )
}

pub fn load_material(
    system: &System,
    layout: &Arc<DescriptorSetLayout>,
    assimp_material: &russimp::material::Material,
    base_dir: &str,
) -> Material {
    let base_textures = assimp_material.textures.get(&TextureType::BaseColor);

    let result_tex = if base_textures.is_some() {
        let assimp_texture = &base_textures.unwrap().first().unwrap();

        let texture = if assimp_texture.path != "" {
            let abs_tex_path = base_dir.to_owned() + assimp_texture.path.as_str();
            load_png_file(&system.queue, &abs_tex_path)
        } else {
            assert_eq!(
                assimp_texture.ach_format_hint, "png",
                "Encompassed texture data should be in png format"
            );

            match assimp_texture
                .data
                .as_ref()
                .expect("Unexpected texture data")
            {
                DataContent::Texel(_) => panic!("Loading textures by texels is not yet supported"),
                DataContent::Bytes(bytes) => load_png(&system.queue, bytes),
            }
        };

        Some(texture)
    } else {
        None
    };

    let uniform_set = match result_tex {
        None => get_descriptor_set(system, layout, load_default_texture(system)),
        Some(x) => get_descriptor_set(system, layout, x),
    };

    Material {
        uniform_set: Some(uniform_set),
    }
}

pub fn load_gltf(system: &System, layout: &Arc<DescriptorSetLayout>, file_path: &Path) -> Mesh {
    let base_dir = file_path.parent().unwrap().to_str().unwrap().to_owned() + "/";
    let scene = Scene::from_file(
        file_path.to_str().unwrap(),
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

    for mesh in &scene.meshes {
        let assimp_vertices = &mesh.vertices;
        let assimp_normals = &mesh.normals;
        let assimp_faces = &mesh.faces;
        let assimp_tex_coords = &mesh.texture_coords;
        let material = load_material(
            system,
            layout,
            scene.materials.get(mesh.material_index as usize).unwrap(),
            &base_dir,
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
        model_matrix: Matrix4::from_scale(1.0),
    }
}

impl Mesh {
    pub fn render(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        general_set: &Arc<PersistentDescriptorSet>,
    ) {
        for mesh_buffer in &self.mesh_buffers {
            let vertex_buffers = (
                mesh_buffer.vertex_buffer.clone(),
                mesh_buffer.normal_buffer.clone(),
                mesh_buffer.tex_coord_buffer.clone(),
            );

            let uniform_set = mesh_buffer.material.uniform_set.as_ref().unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    vec![general_set.clone(), uniform_set.clone()],
                )
                .bind_vertex_buffers(0, vertex_buffers)
                .bind_index_buffer(mesh_buffer.index_buffer.clone())
                .draw_indexed(mesh_buffer.index_buffer.len() as u32, 1, 0, 0, 0)
                .unwrap();
        }
    }
}
