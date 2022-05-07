// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use std::{time::Instant};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    pipeline::{
        Pipeline, PipelineBindPoint,
    },
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow},
};

mod atlas_core;
mod teapot;
use crate::teapot::{VERTICES, NORMALS, INDICES};

fn main() {    
    let system = atlas_core::init("Atlas Engine");
    let device = system.device;
    let mut swapchain = system.swapchain;
    let images = system.images;
    let surface = system.surface;
    let queue = system.queue;

    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, VERTICES)
            .unwrap();
    let normals_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, NORMALS).unwrap();
    let index_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, INDICES).unwrap();

    let uniform_buffer = CpuBufferPool::<vs_mod::ty::Data>::new(device.clone(), BufferUsage::all());

    let vs = vs_mod::load(device.clone()).unwrap();
    let fs = fs_mod::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
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
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let (mut pipeline, mut framebuffers) =
    atlas_core::window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let rotation_start = Instant::now();


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
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

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
                    let scale = Matrix4::from_scale(0.01);
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
                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                    )
                    .unwrap()
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
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
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
