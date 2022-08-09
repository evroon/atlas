use std::sync::Arc;

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SubpassContents},
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage},
    memory::pool::{PotentialDedicatedAllocation, StdMemoryPoolAlloc},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};

use crate::atlas_core::{
    mesh::{Normal, TexCoord, Vertex},
    system::System,
};

pub struct ShadowMapRenderPass {
    pub render_pass: Arc<RenderPass>,
    pub sub_pass: Subpass,
    pub framebuffer: Arc<Framebuffer>,
    pub shadow_map_buffer:
        Arc<ImageView<AttachmentImage<PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>>,
    pub pipeline: Arc<GraphicsPipeline>,
}

pub fn init_render_pass(system: &mut System) -> ShadowMapRenderPass {
    let render_pass = vulkano::ordered_passes_renderpass!(
        system.device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: system.swapchain.image_format(),
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
            {
                color: [final_color],
                depth_stencil: {depth},
                input: []
            }
        ]
    )
    .unwrap();

    let sub_pass = Subpass::from(render_pass.clone(), 0).unwrap();

    let (framebuffer, shadow_map_buffer) = image_setup(
        system.device.clone(),
        render_pass.clone(),
        &mut system.viewport,
    );
    let pipeline = init_pipeline(&system.device, &render_pass);

    ShadowMapRenderPass {
        render_pass,
        sub_pass,
        framebuffer,
        shadow_map_buffer,
        pipeline,
    }
}

pub fn init_pipeline(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
    let shadow_map_vert = shadow_map_vert_mod::load(device.clone()).unwrap();
    let shadow_map_frag = shadow_map_frag_mod::load(device.clone()).unwrap();

    let shadow_map_pass = Subpass::from(render_pass.clone(), 0).unwrap();

    let vertex_input_state = BuffersDefinition::new()
        .vertex::<Vertex>()
        .vertex::<Normal>()
        .vertex::<TexCoord>();

    let shadow_map_pipeline = GraphicsPipeline::start()
        .vertex_input_state(vertex_input_state)
        .vertex_shader(shadow_map_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(shadow_map_frag.entry_point("main").unwrap(), ())
        .color_blend_state(
            ColorBlendState::new(shadow_map_pass.num_color_attachments()).blend_alpha(),
        )
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(shadow_map_pass)
        .build(device.clone())
        .unwrap();

    shadow_map_pipeline
}

pub fn image_setup(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> (Arc<Framebuffer>, Arc<ImageView<AttachmentImage>>) {
    let dimensions = [3000, 2000];
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let color_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::B8G8R8A8_SRGB,
        )
        .unwrap(),
    )
    .unwrap();

    let shadow_map_buffer = ImageView::new_default(
        AttachmentImage::sampled(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![color_buffer.clone(), shadow_map_buffer.clone()],
            ..Default::default()
        },
    )
    .unwrap();

    (framebuffer, shadow_map_buffer)
}

impl ShadowMapRenderPass {
    pub fn prepare_shadow_map_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        viewport: &Viewport,
    ) {
        let clear_values = vec![
            [0.0, 0.0, 0.0, 0.0].into(),
            1f32.into(),
        ];

        builder
            .begin_render_pass(
                self.framebuffer.clone(),
                SubpassContents::Inline,
                clear_values,
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(self.pipeline.clone());
    }
}

pub mod shadow_map_vert_mod {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/shadow_map.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod shadow_map_frag_mod {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/shadow_map.frag",
        types_meta: {
            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
