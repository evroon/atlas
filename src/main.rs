// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use atlas_core::mesh::load_gltf;
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use egui_vulkano::UpdateTexturesResult;
use std::{time::Instant, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    pipeline::{
        Pipeline, PipelineBindPoint, graphics::viewport::Viewport,
    },
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture, FenceSignalFuture}, render_pass::Subpass, device::Device,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow},
};

mod atlas_core;

pub enum FrameEndFuture<F: GpuFuture + 'static> {
    FenceSignalFuture(FenceSignalFuture<F>),
    BoxedFuture(Box<dyn GpuFuture>),
}

impl<F: GpuFuture> FrameEndFuture<F> {
    pub fn now(device: Arc<Device>) -> Self {
        Self::BoxedFuture(sync::now(device).boxed())
    }

    pub fn get(self) -> Box<dyn GpuFuture> {
        match self {
            FrameEndFuture::FenceSignalFuture(f) => f.boxed(),
            FrameEndFuture::BoxedFuture(f) => f,
        }
    }
}

impl<F: GpuFuture> AsMut<dyn GpuFuture> for FrameEndFuture<F> {
    fn as_mut(&mut self) -> &mut (dyn GpuFuture + 'static) {
        match self {
            FrameEndFuture::FenceSignalFuture(f) => f,
            FrameEndFuture::BoxedFuture(f) => f,
        }
    }
}


fn main() {    
    let system = atlas_core::init("Atlas Engine");
    let device = system.device;
    let mut swapchain = system.swapchain;
    let images = system.images;
    let surface = system.surface;
    let queue = system.queue;

    let (vertices, normals, faces) = load_gltf();

    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices)
            .unwrap();
    let normals_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, normals).unwrap();
    let index_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, faces).unwrap();

    let uniform_buffer = CpuBufferPool::<vs_mod::ty::Data>::new(device.clone(), BufferUsage::all());

    let vs = vs_mod::load(device.clone()).unwrap();
    let fs = fs_mod::load(device.clone()).unwrap();

    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
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
            { color: [color], depth_stencil: {depth}, input: [] }, // default renderpass
            { color: [color], depth_stencil: {}, input: [] } // egui renderpass
        ]
    )
    .unwrap();
    
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let (mut pipeline, mut framebuffers) =
    atlas_core::window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(FrameEndFuture::now(device.clone()));
    let rotation_start = Instant::now();
    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(4096, &surface.window());

    let mut egui_painter = egui_vulkano::Painter::new(
        device.clone(),
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).expect("Could not create egui subpass"),
    )
    .expect("Could not create egui painter");

    system.event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent { event, .. } => {
                let egui_consumed_event = egui_winit.on_event(&egui_ctx, &event);
                if !egui_consumed_event {
                    // TODO
                };
            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().as_mut().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: surface.window().inner_size().into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    let (new_pipeline, new_framebuffers) = atlas_core::window_size_dependent_setup(
                        device.clone(),
                        &vs,
                        &fs,
                        &new_images,
                        render_pass.clone(),
                        &mut viewport
                    );
                    pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed();
                    let rotation =
                        elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                    let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                    let aspect_ratio =
                        swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
                    let proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect_ratio,
                        0.01,
                        100.0,
                    );
                    let view = Matrix4::look_at_rh(
                        Point3::new(0.3, 0.3, 1.0),
                        Point3::new(0.0, 0.0, 0.0),
                        Vector3::new(0.0, -1.0, 0.0),
                    );
                    let scale = Matrix4::from_scale(0.25);
                    let view_scaled = view * scale;
                    let world = Matrix4::from(rotation);

                    let uniform_data = vs_mod::ty::Data {
                        world_view: (view_scaled * world).into(),
                        world: world.into(),
                        view: view_scaled.into(),
                        proj: proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let layout = pipeline.layout().set_layouts().get(0).unwrap();
                let set = PersistentDescriptorSet::new(
                    layout.clone(),
                    [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
                )
                .unwrap();

                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                egui_ctx.begin_frame(egui_winit.take_egui_input(surface.window()));

                egui::Window::new("Settings").show(&egui_ctx, |ui| {
                    egui_ctx.settings_ui(ui);
                });

                // Get the shapes from egui
                let egui_output = egui_ctx.end_frame();
                let platform_output = egui_output.platform_output;
                egui_winit.handle_platform_output(surface.window(), &egui_ctx, platform_output);

                let result = egui_painter
                    .update_textures(egui_output.textures_delta, &mut builder)
                    .expect("egui texture error");

                let wait_for_last_frame = result == UpdateTexturesResult::Changed;

                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .bind_vertex_buffers(0, (vertex_buffer.clone(), normals_buffer.clone()))
                    .bind_index_buffer(index_buffer.clone())
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                    .unwrap();


                let size = surface.window().inner_size();
                let sf: f32 = surface.window().scale_factor() as f32;
                egui_painter
                    .draw(
                        &mut builder,
                        [(size.width as f32) / sf, (size.height as f32) / sf],
                        &egui_ctx,
                        egui_output.shapes,
                    )
                    .unwrap();

                builder.end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();

                if wait_for_last_frame {
                    if let Some(FrameEndFuture::FenceSignalFuture(ref mut f)) = previous_frame_end {
                        f.wait(None).unwrap();
                    }
                }

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .get()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(FrameEndFuture::FenceSignalFuture(future));
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(FrameEndFuture::now(device.clone()));
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(FrameEndFuture::now(device.clone()));
                    }
                }
            }
            _ => (),
        }
    });
}


mod vs_mod {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/identity.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod fs_mod {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/identity.frag"
    }
}
