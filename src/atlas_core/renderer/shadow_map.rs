use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage},
    memory::pool::{PotentialDedicatedAllocation, StdMemoryPoolAlloc},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};

use crate::atlas_core::System;

pub struct ShadowMapRenderPass {
    pub render_pass: Arc<RenderPass>,
    pub sub_pass: Subpass,
    pub framebuffer: Arc<Framebuffer>,
    pub shadow_map_buffer:
        Arc<ImageView<AttachmentImage<PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>>,
}

pub fn init_render_pass(system: &mut System) -> ShadowMapRenderPass {
    let render_pass = vulkano::ordered_passes_renderpass!(
        system.device.clone(),
        attachments: {
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        passes: [
            {
                color: [],
                depth_stencil: {depth},
                input: []
            }
        ]
    )
    .unwrap();

    let sub_pass = Subpass::from(render_pass.clone(), 0).unwrap();

    let (framebuffer, shadow_map_buffer) = window_size_dependent_setup(
        system.device.clone(),
        render_pass.clone(),
        &mut system.viewport,
    );

    ShadowMapRenderPass {
        render_pass,
        sub_pass,
        framebuffer,
        shadow_map_buffer,
    }
}

pub fn window_size_dependent_setup(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> (Arc<Framebuffer>, Arc<ImageView<AttachmentImage>>) {
    let dimensions = [1024, 1024];
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let shadow_map_buffer = ImageView::new_default(
        AttachmentImage::sampled(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![shadow_map_buffer.clone()],
            ..Default::default()
        },
    )
    .unwrap();

    (framebuffer, shadow_map_buffer)
}
