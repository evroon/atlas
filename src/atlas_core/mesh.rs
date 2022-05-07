use bytemuck::{Pod, Zeroable};
use russimp::{scene::{Scene, PostProcess}};
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
pub struct Face {
    pub indices: [u32; 3],
}

impl_vertex!(Face, indices);

pub fn load_gltf() -> (Vec<Vertex>, Vec<Normal>, Vec<u32>) {
    let scene = Scene::from_file("assets/models/monkey.glb",
        vec![PostProcess::CalculateTangentSpace,
            PostProcess::Triangulate,
            PostProcess::JoinIdenticalVertices,
            PostProcess::SortByPrimitiveType]).expect("Could not load model");

    let assimp_vertices = &scene.meshes[0].vertices;
    let assimp_normals = &scene.meshes[0].normals;
    let assimp_faces = &scene.meshes[0].faces;

    let vertices = assimp_vertices.iter().map(|v| Vertex {position: [v.x, v.y, v.z]}).collect();
    let normals = assimp_normals.iter().map(|v| Normal {normal: [v.x, v.y, v.z]}).collect();
    let indices: Vec<u32> = assimp_faces.iter().map(|f| [f.0[0], f.0[1], f.0[2]]).flatten().collect();

    (vertices, normals, indices)
}
