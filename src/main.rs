use crate::atlas_core::camera::CameraInputLogic;
use atlas_core::{
    acquire_image,
    camera::construct_camera,
    egui::{get_egui_context, render_egui, update_textures_egui, FrameEndFuture},
    mesh::load_gltf,
    renderer::{
        deferred::{self, deferred_vert_mod, prepare_deferred_pass},
        shadow_map,
        triangle_draw_system::TriangleDrawSystem,
    },
    start_builder, PerformanceInfo,
};
use cgmath::Matrix4;

use std::{path::Path, time::Instant};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
    command_buffer::SubpassContents,
    descriptor_set::PersistentDescriptorSet,
    pipeline::{Pipeline, PipelineBindPoint},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
};
use winit_input_helper::WinitInputHelper;

mod atlas_core;

fn main() {
    let (mut system, event_loop) = atlas_core::init("Atlas Engine");

    let mut deferred_render_pass = deferred::init_render_pass(&mut system);
    let shadow_map_render_pass = shadow_map::init_render_pass(&mut system);

    let uniform_buffer = CpuBufferPool::<deferred_vert_mod::ty::CameraData>::new(
        system.device.clone(),
        BufferUsage::all(),
    );

    let mut recreate_swapchain = false;

    let (egui_ctx, mut egui_winit, mut egui_painter) =
        get_egui_context(&system, &deferred_render_pass.render_pass);

    let mut camera = construct_camera();
    let mut input = WinitInputHelper::new();

    let mut performance_info = PerformanceInfo {
        game_start: Instant::now(),
        last_update: Instant::now(),
        delta_time_ms: 0.0,
    };

    let triangle_system = TriangleDrawSystem::new(&system.queue);

    let layout = deferred_render_pass
        .deferred_pipeline
        .layout()
        .set_layouts()
        .get(1)
        .unwrap();

    let mut mesh = load_gltf(
        &system,
        layout,
        Path::new("assets/models/sponza/sponza.glb"),
    );
    // We need to turn the model upside-down.
    mesh.model_matrix = Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0);

    event_loop.run(move |event, _, control_flow| {
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
                system
                    .previous_frame_end
                    .as_mut()
                    .unwrap()
                    .as_mut()
                    .cleanup_finished();

                deferred::handle_recreate_swapchain(
                    &mut system,
                    &mut deferred_render_pass,
                    &mut recreate_swapchain,
                );

                performance_info.delta_time_ms =
                    (Instant::now() - performance_info.last_update).as_secs_f32() * 1000.0;
                performance_info.last_update = Instant::now();

                let uniform_buffer_subbuffer =
                    camera.get_uniform_buffer(&system, &uniform_buffer, mesh.model_matrix);

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
                    &mut deferred_render_pass.params,
                );

                prepare_deferred_pass(
                    &mut builder,
                    &deferred_render_pass,
                    &system.viewport,
                    image_num,
                );

                let (deferred_set, lighting_set) = deferred::get_layouts(
                    &system,
                    &deferred_render_pass,
                    &shadow_map_render_pass,
                    uniform_buffer_subbuffer,
                );

                mesh.render(
                    &mut builder,
                    &deferred_render_pass.deferred_pipeline,
                    &deferred_set,
                );

                builder
                    .next_subpass(SubpassContents::Inline)
                    .unwrap()
                    .bind_pipeline_graphics(deferred_render_pass.lighting_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        deferred_render_pass.lighting_pipeline.layout().clone(),
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
                    if let Some(FrameEndFuture::FenceSignalFuture(ref mut f)) =
                        system.previous_frame_end
                    {
                        f.wait(None).unwrap();
                    }
                }

                system.finish_frame(
                    command_buffer,
                    &mut recreate_swapchain,
                    acquire_future,
                    image_num,
                )
            }
            _ => (),
        }
    });
}
