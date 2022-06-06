use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    render_pass::{RenderPass, Subpass},
};

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
