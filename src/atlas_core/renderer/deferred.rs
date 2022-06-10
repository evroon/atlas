use std::{f32::consts::PI, sync::Arc};

use cgmath::Vector4;
use vulkano::{
    buffer::{cpu_pool::CpuBufferPoolSubbuffer, BufferUsage, CpuBufferPool},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    memory::pool::{PotentialDedicatedAllocation, StdMemoryPool, StdMemoryPoolAlloc},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{SwapchainCreateInfo, SwapchainCreationError},
};

use winit::window::Window;

use crate::atlas_core::{
    mesh::{Normal, TexCoord, Vertex, Vertex2D},
    texture::get_default_sampler,
    System,
};

use self::{deferred_vert_mod::ty::CameraData, lighting_frag_mod::ty::LightingData};

use super::{shadow_map::ShadowMapRenderPass, triangle_draw_system::TriangleDrawSystem};

#[derive(PartialEq, Clone, Copy)]
pub enum DebugPreviewBuffer {
    FinalOutput = 0,
    Albedo = 1,
    Normal = 2,
    Position = 3,
}

impl DebugPreviewBuffer {
    pub fn get_text(&self) -> &str {
        match self {
            DebugPreviewBuffer::FinalOutput => "Final Output",
            DebugPreviewBuffer::Albedo => "Albedo",
            DebugPreviewBuffer::Normal => "Normal",
            DebugPreviewBuffer::Position => "Position",
        }
    }
}

pub struct RendererParams {
    pub ambient_color: [f32; 4],
    pub directional_direction: [f32; 4],
    pub directional_color: [f32; 4],
    pub preview_buffer: DebugPreviewBuffer,
}

pub struct DeferredRenderPass {
    pub render_pass: Arc<RenderPass>,
    pub deferred_pass: Subpass,
    pub lighting_pass: Subpass,
    pub params: RendererParams,

    pub deferred_framebuffers: Vec<Arc<Framebuffer>>,

    pub deferred_pipeline: Arc<GraphicsPipeline>,
    pub lighting_pipeline: Arc<GraphicsPipeline>,

    pub color_buffer:
        Arc<ImageView<AttachmentImage<PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>>,
    pub normal_buffer:
        Arc<ImageView<AttachmentImage<PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>>,
    pub position_buffer:
        Arc<ImageView<AttachmentImage<PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>>,
}

pub fn get_default_params() -> RendererParams {
    RendererParams {
        ambient_color: Vector4 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 0.3,
        }
        .into(),
        directional_direction: Vector4 {
            x: 0.0,
            y: -PI / 2.0,
            z: -PI / 2.0,
            w: 0.0,
        }
        .into(),
        directional_color: Vector4 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 0.3,
        }
        .into(),
        preview_buffer: DebugPreviewBuffer::FinalOutput,
    }
}

pub fn get_lighting_uniform_buffer(
    device: &Arc<Device>,
    params: &RendererParams,
) -> Arc<CpuBufferPoolSubbuffer<LightingData, Arc<StdMemoryPool>>> {
    let lighting_buffer = CpuBufferPool::<lighting_frag_mod::ty::LightingData>::new(
        device.clone(),
        BufferUsage::all(),
    );

    let uniform_data = lighting_frag_mod::ty::LightingData {
        ambient_color: params.ambient_color,
        directional_direction: params.directional_direction,
        directional_color: params.directional_color,
        preview_type: params.preview_buffer as i32,
    };

    lighting_buffer.next(uniform_data).unwrap()
}

pub fn init_render_pass(system: &mut System) -> DeferredRenderPass {
    let render_pass = vulkano::ordered_passes_renderpass!(
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
            positions: {
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
                color: [albedo, normals, positions],
                depth_stencil: {depth},
                input: []
            },
            // Apply lighting by reading these three attachments and writing to `final_color`.
            {
                color: [final_color],
                depth_stencil: {},
                input: [albedo, normals, positions] //, depth
            },
            // egui renderpass
            { color: [final_color], depth_stencil: {}, input: [] }
        ]
    )
    .unwrap();

    let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
    let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

    let (deferred_framebuffers, color_buffer, normal_buffer, position_buffer) =
        window_size_dependent_setup(
            system.device.clone(),
            &system.images,
            render_pass.clone(),
            &mut system.viewport,
        );

    let (deferred_pipeline, lighting_pipeline) = init_pipelines(&system.device, &render_pass);

    DeferredRenderPass {
        deferred_framebuffers,
        color_buffer,
        normal_buffer,
        position_buffer,
        render_pass,
        deferred_pass,
        lighting_pass,
        deferred_pipeline,
        lighting_pipeline,
        params: get_default_params(),
    }
}

pub fn init_pipelines(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
) -> (Arc<GraphicsPipeline>, Arc<GraphicsPipeline>) {
    let deferred_vert = deferred_vert_mod::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag_mod::load(device.clone()).unwrap();
    let lighting_vert = lighting_vert_mod::load(device.clone()).unwrap();
    let lighting_frag = lighting_frag_mod::load(device.clone()).unwrap();

    let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
    let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

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
        .render_pass(deferred_pass)
        .build(device.clone())
        .unwrap();

    let lighting_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex2D>())
        .vertex_shader(lighting_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(lighting_frag.entry_point("main").unwrap(), ())
        .render_pass(lighting_pass)
        .build(device.clone())
        .unwrap();

    (deferred_pipeline, lighting_pipeline)
}

pub fn get_layouts(
    system: &System,
    deferred_render_pass: &DeferredRenderPass,
    shadow_map_render_pass: &ShadowMapRenderPass,
    uniform_buffer_subbuffer: Arc<CpuBufferPoolSubbuffer<CameraData, Arc<StdMemoryPool>>>,
) -> (Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>) {
    let deferred_layout = deferred_render_pass
        .deferred_pipeline
        .layout()
        .set_layouts()
        .get(0)
        .unwrap();
    let deferred_set = PersistentDescriptorSet::new(
        deferred_layout.clone(),
        [WriteDescriptorSet::buffer(
            0,
            uniform_buffer_subbuffer.clone(),
        )],
    )
    .unwrap();

    let lighting_layout = deferred_render_pass
        .lighting_pipeline
        .layout()
        .set_layouts()
        .get(0)
        .unwrap();
    let lighting_set = PersistentDescriptorSet::new(
        lighting_layout.clone(),
        [
            WriteDescriptorSet::image_view(0, deferred_render_pass.color_buffer.clone()),
            WriteDescriptorSet::image_view(1, deferred_render_pass.normal_buffer.clone()),
            WriteDescriptorSet::image_view(2, deferred_render_pass.position_buffer.clone()),
            WriteDescriptorSet::image_view_sampler(
                3,
                shadow_map_render_pass.shadow_map_buffer.clone(),
                get_default_sampler(&system.device).clone(),
            ),
            WriteDescriptorSet::buffer(
                10,
                get_lighting_uniform_buffer(&system.device.clone(), &deferred_render_pass.params),
            ),
        ],
    )
    .unwrap();
    (deferred_set, lighting_set)
}

impl DeferredRenderPass {
    pub fn prepare_deferred_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        viewport: &Viewport,
        image_num: usize,
    ) {
        let clear_values = vec![
            [0.0, 0.0, 0.0, 1.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
            1f32.into(),
        ];

        builder
            .begin_render_pass(
                self.deferred_framebuffers[image_num].clone(),
                SubpassContents::Inline,
                clear_values,
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(self.deferred_pipeline.clone());
    }

    pub fn prepare_lighting_subpass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        lighting_set: Arc<PersistentDescriptorSet>,
        triangle_system: &TriangleDrawSystem,
    ) {
        builder
            .next_subpass(SubpassContents::Inline)
            .unwrap()
            .bind_pipeline_graphics(self.lighting_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.lighting_pipeline.layout().clone(),
                0,
                lighting_set.clone(),
            )
            .bind_vertex_buffers(0, triangle_system.vertex_buffer.clone())
            .draw(6, 1, 0, 0)
            .unwrap();
    }

    pub fn handle_recreate_swapchain(&mut self, system: &mut System) {
        if system.recreate_swapchain {
            let (new_swapchain, new_images) = match system.swapchain.recreate(SwapchainCreateInfo {
                image_extent: system.surface.window().inner_size().into(),
                ..system.swapchain.create_info()
            }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

            system.swapchain = new_swapchain;
            let (new_framebuffers, new_color_buffer, new_normal_buffer, new_position_buffer) =
                window_size_dependent_setup(
                    system.device.clone(),
                    &new_images,
                    self.render_pass.clone(),
                    &mut system.viewport,
                );

            self.deferred_framebuffers = new_framebuffers;
            self.color_buffer = new_color_buffer;
            self.normal_buffer = new_normal_buffer;
            self.position_buffer = new_position_buffer;

            system.recreate_swapchain = false;
        }
    }
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
        path: "src/shaders/deferred.frag",
        types_meta: {
            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod lighting_vert_mod {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/lighting.vert",
        types_meta: {
            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod lighting_frag_mod {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/lighting.frag",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> (
    Vec<Arc<Framebuffer>>,
    Arc<ImageView<AttachmentImage>>,
    Arc<ImageView<AttachmentImage>>,
    Arc<ImageView<AttachmentImage>>,
) {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();
    let color_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::A2B10G10R10_UNORM_PACK32,
        )
        .unwrap(),
    )
    .unwrap();
    let normal_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::R16G16B16A16_SFLOAT,
        )
        .unwrap(),
    )
    .unwrap();
    let position_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::R16G16B16A16_SFLOAT,
        )
        .unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        color_buffer.clone(),
                        normal_buffer.clone(),
                        position_buffer.clone(),
                        depth_buffer.clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    (framebuffers, color_buffer, normal_buffer, position_buffer)
}
