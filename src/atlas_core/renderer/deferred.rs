use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState, depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState, vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline,
    },
    render_pass::{RenderPass, Subpass},
    swapchain::Swapchain,
};
use winit::window::Window;

use crate::atlas_core::mesh::{Normal, TexCoord, Vertex, Vertex2D};

pub struct DeferredRenderPass {
    pub render_pass: Arc<RenderPass>,
    pub deferred_pass: Subpass,
    pub lighting_pass: Subpass,
}

pub fn init_render_pass(
    device: &Arc<Device>,
    swapchain: &Arc<Swapchain<Window>>,
) -> DeferredRenderPass {
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
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
            // Deferred pass. Write to the diffuse, normals and depth attachments.
            {
                color: [albedo, normals],
                depth_stencil: {depth},
                input: []
            },
            // Apply lighting by reading these three attachments and writing to `final_color`.
            {
                color: [final_color],
                depth_stencil: {},
                input: [albedo, normals] //, depth
            },
            // egui renderpass
            { color: [final_color], depth_stencil: {}, input: [] }
        ]
    )
    .unwrap();

    let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
    let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

    DeferredRenderPass {
        render_pass,
        deferred_pass,
        lighting_pass,
    }
}

pub fn init_pipelines(
    device: &Arc<Device>,
    render_pass: &DeferredRenderPass,
) -> (Arc<GraphicsPipeline>, Arc<GraphicsPipeline>) {
    let deferred_vert = deferred_vert_mod::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag_mod::load(device.clone()).unwrap();
    let lighting_vert = lighting_vert_mod::load(device.clone()).unwrap();
    let lighting_frag = lighting_frag_mod::load(device.clone()).unwrap();

    let deferred_pass = Subpass::from(render_pass.render_pass.clone(), 0).unwrap();
    let lighting_pass = Subpass::from(render_pass.render_pass.clone(), 1).unwrap();

    let vertex_input_state = BuffersDefinition::new()
        .vertex::<Vertex>()
        .vertex::<Normal>()
        .vertex::<TexCoord>();

    let deferred_pipeline = GraphicsPipeline::start()
        .vertex_input_state(vertex_input_state)
        .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
        .color_blend_state(
            ColorBlendState::new(deferred_pass.num_color_attachments()).blend_alpha(),
        )
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        // .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(deferred_pass)
        .build(device.clone())
        .unwrap();

    let lighting_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex2D>())
        .vertex_shader(lighting_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(lighting_frag.entry_point("main").unwrap(), ())
        // .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_pass)
        .build(device.clone())
        .unwrap();

    (deferred_pipeline, lighting_pipeline)
}

pub mod deferred_vert_mod {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/deferred.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod deferred_frag_mod {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/deferred.frag"
    }
}

mod lighting_vert_mod {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/lighting.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod lighting_frag_mod {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/lighting.frag"
    }
}
