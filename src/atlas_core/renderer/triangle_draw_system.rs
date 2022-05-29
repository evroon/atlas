use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Queue,
};

use crate::atlas_core::mesh::Vertex2D;

pub struct TriangleDrawSystem {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex2D]>>,
}

impl TriangleDrawSystem {
    pub fn new(gfx_queue: &Arc<Queue>) -> TriangleDrawSystem {
        let vertices = [
            Vertex2D {
                position: [1.0, -1.0],
            },
            Vertex2D {
                position: [-1.0, -1.0],
            },
            Vertex2D {
                position: [1.0, 1.0],
            },
            Vertex2D {
                position: [-1.0, -1.0],
            },
            Vertex2D {
                position: [-1.0, 1.0],
            },
            Vertex2D {
                position: [1.0, 1.0],
            },
        ];
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage::all(),
                false,
                vertices,
            )
            .expect("failed to create triangle buffer")
        };

        TriangleDrawSystem { vertex_buffer }
    }
}
