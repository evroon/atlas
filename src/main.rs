use crate::atlas_core::camera::CameraInputLogic;
use atlas_core::{
    camera::construct_camera,
    egui::{get_egui_context, render_egui, update_textures_egui, FrameEndFuture},
    mesh::load_gltf,
    renderer::{
        deferred::{self, deferred_vert_mod, get_lighting_uniform_buffer},
        triangle_draw_system::TriangleDrawSystem,
    },
    PerformanceInfo,
};

use std::{path::Path, time::Instant};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    pipeline::{graphics::viewport::Viewport, Pipeline, PipelineBindPoint},
    swapchain::{acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError},
    sync::{FlushError, GpuFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
};
use winit_input_helper::WinitInputHelper;

mod atlas_core;

fn main() {
    let mut system = atlas_core::init("Atlas Engine");
    let uniform_buffer = CpuBufferPool::<deferred_vert_mod::ty::Data>::new(
        system.device.clone(),
        BufferUsage::all(),
    );

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let (mut framebuffers, mut color_buffer, mut normal_buffer, mut position_buffer) =
        atlas_core::window_size_dependent_setup(
            system.device.clone(),
            &system.images,
            system.render_pass.render_pass.clone(),
            &mut viewport,
        );

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(FrameEndFuture::now(system.device.clone()));

    let (egui_ctx, mut egui_winit, mut egui_painter) =
        get_egui_context(&system, &system.render_pass.render_pass);

    let mut camera = construct_camera();
    let mut input = WinitInputHelper::new();

    let game_start = Instant::now();
    let mut last_update = Instant::now();

    let mut performance_info = PerformanceInfo {
        game_start,
        delta_time_ms: 0.0,
    };

    let (deferred_pipeline, lighting_pipeline) =
        deferred::init_pipelines(&system.device, &system.render_pass);

    let triangle_system = TriangleDrawSystem::new(&system.queue);

    let layout = deferred_pipeline.layout().set_layouts().get(1).unwrap();
    let mesh = load_gltf(
        &system,
        layout,
        Path::new("assets/models/sponza/sponza.glb"),
    );

    system.event_loop.run(move |event, _, control_flow| {
        if input.update(&event) {
            camera.handle_event(&input);
        }

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
                egui_winit.on_event(&egui_ctx, &event);
            }
            Event::RedrawEventsCleared => {
                previous_frame_end
                    .as_mut()
                    .unwrap()
                    .as_mut()
                    .cleanup_finished();

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
                    let (
                        new_framebuffers,
                        new_color_buffer,
                        new_normal_buffer,
                        new_position_buffer,
                    ) = atlas_core::window_size_dependent_setup(
                        system.device.clone(),
                        &new_images,
                        system.render_pass.render_pass.clone(),
                        &mut viewport,
                    );

                    framebuffers = new_framebuffers;
                    color_buffer = new_color_buffer;
                    normal_buffer = new_normal_buffer;
                    position_buffer = new_position_buffer;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    performance_info.delta_time_ms =
                        (Instant::now() - last_update).as_secs_f32() * 1000.0;
                    last_update = Instant::now();

                    let extent = system.swapchain.image_extent();
                    camera.aspect_ratio = extent[0] as f32 / extent[1] as f32;
                    camera.update();

                    let uniform_data = deferred_vert_mod::ty::Data {
                        world_view: camera.world_view.into(),
                        world: camera.world.into(),
                        view: camera.view.into(),
                        proj: camera.proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let deferred_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
                let deferred_set = PersistentDescriptorSet::new(
                    deferred_layout.clone(),
                    [WriteDescriptorSet::buffer(
                        0,
                        uniform_buffer_subbuffer.clone(),
                    )],
                )
                .unwrap();

                let lighting_layout = lighting_pipeline.layout().set_layouts().get(0).unwrap();
                let lighting_set = PersistentDescriptorSet::new(
                    lighting_layout.clone(),
                    [
                        WriteDescriptorSet::image_view(0, color_buffer.clone()),
                        WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                        WriteDescriptorSet::image_view(2, position_buffer.clone()),
                        WriteDescriptorSet::buffer(
                            3,
                            get_lighting_uniform_buffer(
                                &system.device.clone(),
                                &system.render_pass.params,
                            ),
                        ),
                    ],
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

                let (shapes, wait_for_last_frame) = update_textures_egui(
                    &performance_info,
                    &system.info,
                    &mut builder,
                    &system.surface,
                    &egui_ctx,
                    &mut egui_painter,
                    &mut egui_winit,
                    &mut system.render_pass.params,
                );

                let clear_values = vec![
                    [0.0, 0.0, 0.0, 1.0].into(),
                    [0.0, 0.0, 0.0, 1.0].into(),
                    [0.0, 0.0, 0.0, 1.0].into(),
                    [0.0, 0.0, 0.0, 1.0].into(),
                    1f32.into(),
                ];

                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(deferred_pipeline.clone());

                mesh.render(&mut builder, &deferred_pipeline, &deferred_set);

                builder
                    .next_subpass(SubpassContents::Inline)
                    .unwrap()
                    .bind_pipeline_graphics(lighting_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        lighting_pipeline.layout().clone(),
                        0,
                        lighting_set.clone(),
                    )
                    .bind_vertex_buffers(0, triangle_system.vertex_buffer.clone())
                    .draw(6, 1, 0, 0)
                    .unwrap();

                render_egui(
                    &mut builder,
                    &system.surface,
                    &egui_ctx,
                    shapes,
                    &mut egui_painter,
                );

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
                    .then_swapchain_present(
                        system.queue.clone(),
                        system.swapchain.clone(),
                        image_num,
                    )
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
