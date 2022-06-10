use crate::atlas_core::camera::CameraInputLogic;
use atlas_core::{
    acquire_image,
    camera::construct_camera,
    egui::get_egui_context,
    mesh::load_gltf,
    renderer::{
        deferred::{self, deferred_vert_mod},
        shadow_map,
    },
    start_builder,
};
use cgmath::Matrix4;

use std::path::Path;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
    descriptor_set::PersistentDescriptorSet,
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

    let mut egui_data = get_egui_context(&system, &deferred_render_pass.render_pass);

    let mut camera = construct_camera();
    let mut input = WinitInputHelper::new();

    let layout = deferred_render_pass.get_deferred_layout();

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
                system.recreate_swapchain = true;
            }
            Event::WindowEvent { event, .. } => {
                egui_data.egui_winit.on_event(&egui_data.egui_ctx, &event);
            }
            Event::RedrawEventsCleared => {
                system.cleanup_finished();

                deferred_render_pass.handle_recreate_swapchain(&mut system);

                system.performance_info.update();

                let uniform_buffer_subbuffer =
                    camera.get_uniform_buffer(&system, &uniform_buffer, mesh.model_matrix);

                let image_update_result =
                    acquire_image(&system.swapchain, &mut system.recreate_swapchain);
                if image_update_result.is_err() {
                    return;
                }

                let (image_num, acquire_future) = image_update_result.unwrap();

                let mut builder = start_builder(&system.device, &system.queue);

                let (shapes, wait_for_last_frame) = egui_data.update_textures_egui(
                    &system,
                    &mut builder,
                    &mut deferred_render_pass.params,
                );

                deferred_render_pass.prepare_deferred_pass(
                    &mut builder,
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

                deferred_render_pass.prepare_lighting_subpass(
                    &mut builder,
                    lighting_set,
                    &system.triangle_system,
                );

                egui_data.render_egui(&mut builder, &system.surface, shapes);

                builder.end_render_pass().unwrap();
                system.finish_frame(
                    builder.build().unwrap(),
                    acquire_future,
                    image_num,
                    wait_for_last_frame,
                )
            }
            _ => (),
        }
    });
}
