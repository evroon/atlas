use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage, SwapchainImage},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};
use winit::window::Window;

pub struct ShadowMapRenderPass {
    pub render_pass: Arc<RenderPass>,
    pub sub_pass: Subpass,
}

pub fn init_render_pass(device: &Arc<Device>) -> ShadowMapRenderPass {
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
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

    ShadowMapRenderPass {
        render_pass,
        sub_pass,
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
