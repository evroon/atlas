use vulkano::format::Format;

use crate::atlas_core::System;

pub fn init_render_pass(system: &System) -> std::sync::Arc<vulkano::render_pass::RenderPass> {
    vulkano::ordered_passes_renderpass!(
        system.device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: system.swapchain.image_format(),
                samples: 1,
            },
            albedo: {
                load: Clear,
                store: DontCare,
                format: Format::A2B10G10R10_UNORM_PACK32,
                samples: 1,
            },
            normals: {
                load: Clear,
                store: DontCare,
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        passes: [
            // Write to the diffuse, normals and depth attachments.
            {
                color: [albedo, normals],
                depth_stencil: {depth},
                input: []
            },
            // Apply lighting by reading these three attachments and writing to `final_color`.
            {
                color: [final_color],
                depth_stencil: {},
                input: [albedo, normals, depth]
            },
            // egui renderpass
            { color: [final_color], depth_stencil: {}, input: [] }
        ]
    )
    .unwrap()
}
