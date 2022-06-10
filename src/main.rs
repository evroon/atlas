use crate::atlas_core::camera::CameraInputLogic;
use atlas_core::{
    acquire_image,
    camera::construct_camera,
    egui::{get_egui_context, render_egui, update_textures_egui, FrameEndFuture},
    mesh::load_gltf,
    renderer::{
        deferred::{self, deferred_vert_mod, get_lighting_uniform_buffer, prepare_deferred_pass},
        shadow_map,
        triangle_draw_system::TriangleDrawSystem,
    },
    start_builder,
    texture::get_default_sampler,
    PerformanceInfo,
};
use cgmath::Matrix4;

use std::{path::Path, time::Instant};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
    command_buffer::SubpassContents,
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    pipeline::{Pipeline, PipelineBindPoint},
    swapchain::{SwapchainCreateInfo, SwapchainCreationError},
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
    let uniform_buffer = CpuBufferPool::<deferred_vert_mod::ty::CameraData>::new(
        system.device.clone(),
        BufferUsage::all(),
    );

    let (mut framebuffers, mut color_buffer, mut normal_buffer, mut position_buffer) =
        deferred::window_size_dependent_setup(
            system.device.clone(),
            &system.images,
            system.deferred_render_pass.render_pass.clone(),
            &mut system.viewport,
        );

    let (shadow_map_framebuffers, shadow_map_buffer) = shadow_map::window_size_dependent_setup(
        system.device.clone(),
        system.shadow_map_render_pass.render_pass.clone(),
        &mut system.viewport,
    );

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(FrameEndFuture::now(system.device.clone()));

    let (egui_ctx, mut egui_winit, mut egui_painter) =
        get_egui_context(&system, &system.deferred_render_pass.render_pass);

    let mut camera = construct_camera();
    let mut input = WinitInputHelper::new();

    let game_start = Instant::now();
    let mut last_update = Instant::now();

    let mut performance_info = PerformanceInfo {
        game_start,
        delta_time_ms: 0.0,
    };

    let (deferred_pipeline, lighting_pipeline) =
        deferred::init_pipelines(&system.device, &system.deferred_render_pass);

    let triangle_system = TriangleDrawSystem::new(&system.queue);

    let layout = deferred_pipeline.layout().set_layouts().get(1).unwrap();
    let mut mesh = load_gltf(
        &system,
        layout,
        Path::new("assets/models/sponza/sponza.glb"),
    );
    // We need to turn the model upside-down.
    mesh.model_matrix = Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0);

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
                    ) = deferred::window_size_dependent_setup(
                        system.device.clone(),
                        &new_images,
                        system.deferred_render_pass.render_pass.clone(),
                        &mut system.viewport,
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
                    camera.world = mesh.model_matrix.into();
                    camera.update();

                    let uniform_data = deferred_vert_mod::ty::CameraData {
                        world_view: camera.world_view.into(),
                        world: camera.world.into(),
                        view: camera.view.into(),
                        proj: camera.proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let image_update_result = acquire_image(&system.swapchain, &mut recreate_swapchain);
                if image_update_result.is_err() {
                    return;
                }

                let (image_num, acquire_future) = image_update_result.unwrap();

                let mut builder = start_builder(&system.device, &system.queue);

                let (shapes, wait_for_last_frame) = update_textures_egui(
                    &performance_info,
                    &system.info,
                    &mut builder,
                    &system.surface,
                    &egui_ctx,
                    &mut egui_painter,
                    &mut egui_winit,
                    &mut system.deferred_render_pass.params,
                );

                prepare_deferred_pass(
                    &mut builder,
                    &framebuffers[image_num],
                    &deferred_pipeline,
                    &system.viewport,
                );

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
