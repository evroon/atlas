use std::sync::Arc;
use std::time::Instant;
use vulkano::device::Features;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};

use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use self::mesh::{Normal, TexCoord, Vertex};

pub mod camera;
pub mod egui;
pub mod mesh;
pub mod texture;

pub struct PerformanceInfo {
    pub game_start: Instant,
    pub delta_time_ms: f32,
}

pub struct SystemInfo {
    pub device_name: String,
    pub device_type: String,
}

pub struct System {
    pub info: SystemInfo,
    pub event_loop: EventLoop<()>,
    pub device: Arc<Device>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub images: Vec<Arc<SwapchainImage<Window>>>,
    pub surface: Arc<Surface<Window>>,
    pub queue: Arc<Queue>,
}

pub fn init(title: &str) -> System {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title(title)
        .with_inner_size(LogicalSize::new(3000.0, 2000.0))
        // .with_fullscreen(Some(Fullscreen::Borderless(None)))
        .build_vk_surface(&event_loop, instance.clone())
        .expect("Failed to create a window");

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    let systtem_properties = physical_device.properties();

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            enabled_features: Features {
                descriptor_indexing: true,
                shader_uniform_buffer_array_non_uniform_indexing: true,
                runtime_descriptor_array: true,
                descriptor_binding_variable_descriptor_count: true,
                ..Features::none()
            },
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format: Some(Format::B8G8R8A8_SRGB),
                image_extent: surface.window().inner_size().into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    System {
        info: SystemInfo {
            device_name: systtem_properties.device_name.clone(),
            device_type: format!("{:?}", systtem_properties.device_type),
        },
        event_loop,
        device,
        swapchain,
        images,
        surface,
        queue,
    }
}

pub fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &ShaderModule,
    fs: &ShaderModule,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>) {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let vertex_input_state = BuffersDefinition::new()
        .vertex::<Vertex>()
        .vertex::<Normal>()
        .vertex::<TexCoord>();

    // let pipeline_layout = {
    //     let mut layout_create_infos: Vec<_> = DescriptorSetLayoutCreateInfo::from_requirements(
    //         fs.entry_point("main").unwrap().descriptor_requirements(),
    //     );

    //     // Set 0, Binding 0
    //     let binding = layout_create_infos[0].bindings.get_mut(&0).unwrap();
    //     binding.variable_descriptor_count = true;
    //     binding.descriptor_count = 1;

    //     let set_layouts = layout_create_infos
    //         .into_iter()
    //         .map(|desc| Ok(DescriptorSetLayout::new(device.clone(), desc.clone())?))
    //         .collect::<Result<Vec<_>, DescriptorSetLayoutCreationError>>()
    //         .unwrap();

    //     PipelineLayout::new(
    //         device.clone(),
    //         PipelineLayoutCreateInfo {
    //             set_layouts,
    //             push_constant_ranges: fs
    //                 .entry_point("main")
    //                 .unwrap()
    //                 .push_constant_requirements()
    //                 .cloned()
    //                 .into_iter()
    //                 .collect(),
    //             ..Default::default()
    //         },
    //     )
    //     .unwrap()
    // };

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(vertex_input_state)
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend_alpha())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(subpass)
        .build(device.clone())
        // .with_pipeline_layout(device.clone(), pipeline_layout)
        .unwrap();

    (pipeline, framebuffers)
}
