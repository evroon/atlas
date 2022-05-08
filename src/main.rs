use atlas_core::{mesh::load_gltf, egui::{get_egui_context, FrameEndFuture, render_egui, update_textures_egui}, camera::construct_camera};
use cgmath::{Matrix3, Rad};
use std::{time::Instant};
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
    sync::{FlushError, GpuFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow},
};

mod atlas_core;


fn main() {    
    let mut system = atlas_core::init("Atlas Engine");

    let mesh = load_gltf(&system);

    let uniform_buffer = CpuBufferPool::<vs_mod::ty::Data>::new(system.device.clone(), BufferUsage::all());

    let vs = vs_mod::load(system.device.clone()).unwrap();
    let fs = fs_mod::load(system.device.clone()).unwrap();

    let render_pass = vulkano::ordered_passes_renderpass!(
        system.device.clone(),
        attachments: {
            color: {
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
    atlas_core::window_size_dependent_setup(system.device.clone(), &vs, &fs, &system.images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(FrameEndFuture::now(system.device.clone()));
    let rotation_start = Instant::now();

    let (egui_ctx, mut egui_winit, mut egui_painter) = get_egui_context(&system, &render_pass);

    let mut camera = construct_camera();

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
                        match system.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: system.surface.window().inner_size().into(),
                            ..system.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                        system.swapchain = new_swapchain;
                    let (new_pipeline, new_framebuffers) = atlas_core::window_size_dependent_setup(
                        system.device.clone(),
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
                    camera.aspect_ratio = system.swapchain.image_extent()[0] as f32 / system.swapchain.image_extent()[1] as f32;
                    camera.update(rotation);

                    let uniform_data = vs_mod::ty::Data {
                        world_view: camera.world_view.into(),
                        world: camera.world.into(),
                        view: camera.view.into(),
                        proj: camera.proj.into(),
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
                    match acquire_next_image(system.swapchain.clone(), None) {
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
                    system.device.clone(),
                    system.queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let (shapes, wait_for_last_frame) = update_textures_egui(&mut builder, &system.surface, &egui_ctx, &mut egui_painter, &mut egui_winit);

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
                    .bind_vertex_buffers(0, (mesh.vertex_buffer.clone(), mesh.normal_buffer.clone()))
                    .bind_index_buffer(mesh.index_buffer.clone())
                    .draw_indexed(mesh.index_buffer.len() as u32, 1, 0, 0, 0)
                    .unwrap();


                render_egui(&mut builder, &system.surface, &egui_ctx, shapes, &mut egui_painter);

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
                    .then_execute(system.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(system.queue.clone(), system.swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(FrameEndFuture::FenceSignalFuture(future));
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(FrameEndFuture::now(system.device.clone()));
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(FrameEndFuture::now(system.device.clone()));
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
