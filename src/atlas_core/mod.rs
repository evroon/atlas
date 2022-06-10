use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::device::Features;
use vulkano::swapchain::{
    acquire_next_image, AcquireError, PresentFuture, Surface, SwapchainAcquireFuture,
};
use vulkano::sync::{FlushError, GpuFuture, JoinFuture};
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    swapchain::{Swapchain, SwapchainCreateInfo},
};

use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use self::egui::FrameEndFuture;

pub mod camera;
pub mod egui;
pub mod mesh;
pub mod renderer;
pub mod texture;

pub struct PerformanceInfo {
    pub game_start: Instant,
    pub last_update: Instant,
    pub delta_time_ms: f32,
}

impl PerformanceInfo {
    pub fn update(&mut self) {
        self.delta_time_ms = (Instant::now() - self.last_update).as_secs_f32() * 1000.0;
        self.last_update = Instant::now();
    }
}

pub struct SystemInfo {
    pub device_name: String,
    pub device_type: String,
}

pub struct System {
    pub info: SystemInfo,
    pub device: Arc<Device>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub images: Vec<Arc<SwapchainImage<Window>>>,
    pub surface: Arc<Surface<Window>>,
    pub queue: Arc<Queue>,
    pub viewport: Viewport,
    pub previous_frame_end: Option<
        FrameEndFuture<
            PresentFuture<
                CommandBufferExecFuture<
                    JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture<Window>>,
                    PrimaryAutoCommandBuffer,
                >,
                Window,
            >,
        >,
    >,
    pub performance_info: PerformanceInfo,
    pub recreate_swapchain: bool,
}

pub fn init(title: &str) -> (System, EventLoop<()>) {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title(title)
        .with_inner_size(LogicalSize::new(3000.0_f32, 2000.0_f32))
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

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    let previous_frame_end = Some(FrameEndFuture::now(device.clone()));

    let performance_info = PerformanceInfo {
        game_start: Instant::now(),
        last_update: Instant::now(),
        delta_time_ms: 0.0,
    };

    (
        System {
            info: SystemInfo {
                device_name: systtem_properties.device_name.clone(),
                device_type: format!("{:?}", systtem_properties.device_type),
            },
            device,
            swapchain,
            images,
            surface,
            queue,
            viewport,
            previous_frame_end,
            performance_info,
            recreate_swapchain: true,
        },
        event_loop,
    )
}

pub fn acquire_image(
    swapchain: &Arc<Swapchain<Window>>,
    recreate_swapchain: &mut bool,
) -> Result<(usize, SwapchainAcquireFuture<Window>), AcquireError> {
    let (image_num, suboptimal, acquire_future) = match acquire_next_image(swapchain.clone(), None)
    {
        Ok(r) => r,
        Err(AcquireError::OutOfDate) => {
            *recreate_swapchain = true;
            return Err(AcquireError::OutOfDate);
        }
        Err(e) => panic!("Failed to acquire next image: {:?}", e),
    };

    if suboptimal {
        *recreate_swapchain = true;
    }

    Ok((image_num, acquire_future))
}

pub fn start_builder(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
    AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap()
}

impl System {
    pub fn finish_frame(
        &mut self,
        command_buffer: PrimaryAutoCommandBuffer,
        acquire_future: SwapchainAcquireFuture<Window>,
        image_num: usize,
        wait_for_last_frame: bool,
    ) {
        if wait_for_last_frame {
            if let Some(FrameEndFuture::FenceSignalFuture(ref mut f)) = self.previous_frame_end {
                f.wait(None).unwrap();
            }
        }

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .get()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(FrameEndFuture::FenceSignalFuture(future));
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(FrameEndFuture::now(self.device.clone()));
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(FrameEndFuture::now(self.device.clone()));
            }
        }
    }

    pub fn cleanup_finished(&mut self) {
        self.previous_frame_end
            .as_mut()
            .unwrap()
            .as_mut()
            .cleanup_finished();
    }
}
