use std::sync::Arc;

use egui::{epaint::ClippedShape, TextStyle};
use egui_vulkano::UpdateTexturesResult;
use egui_winit::State;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::Device,
    render_pass::{RenderPass, Subpass},
    swapchain::Surface,
    sync::{self, FenceSignalFuture, GpuFuture},
};
use winit::window::Window;

use super::System;

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

pub fn get_egui_context(
    system: &System,
    render_pass: &Arc<RenderPass>,
) -> (egui::Context, State, egui_vulkano::Painter) {
    let egui_ctx = egui::Context::default();

    // Increase text size
    let mut style: egui::Style = (*egui_ctx.style()).clone();
    style.text_styles.get_mut(&TextStyle::Button).unwrap().size = 19.0;
    style.text_styles.get_mut(&TextStyle::Body).unwrap().size = 19.0;
    egui_ctx.set_style(style);

    let egui_winit = egui_winit::State::new(4096, &system.surface.window());

    let egui_painter = egui_vulkano::Painter::new(
        system.device.clone(),
        system.queue.clone(),
        Subpass::from(render_pass.clone(), 1).expect("Could not create egui subpass"),
    )
    .expect("Could not create egui painter");

    (egui_ctx, egui_winit, egui_painter)
}

pub fn update_textures_egui(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    surface: &Arc<Surface<Window>>,
    egui_ctx: &egui::Context,
    egui_painter: &mut egui_vulkano::Painter,
    egui_winit: &mut State,
) -> (Vec<ClippedShape>, bool) {
    egui_ctx.begin_frame(egui_winit.take_egui_input(surface.window()));

    egui::Window::new("Settings").show(&egui_ctx, |ui| {
        egui_ctx.settings_ui(ui);
    });

    // Get the shapes from egui
    let egui_output = egui_ctx.end_frame();
    let platform_output = egui_output.platform_output;
    egui_winit.handle_platform_output(surface.window(), &egui_ctx, platform_output);

    let result = egui_painter
        .update_textures(egui_output.textures_delta, builder)
        .expect("egui texture error");

    let wait_for_last_frame = result == UpdateTexturesResult::Changed;
    (egui_output.shapes, wait_for_last_frame)
}

pub fn render_egui(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    surface: &Arc<Surface<Window>>,
    egui_ctx: &egui::Context,
    shapes: Vec<ClippedShape>,
    egui_painter: &mut egui_vulkano::Painter,
) {
    let size = surface.window().inner_size();
    let sf: f32 = surface.window().scale_factor() as f32;
    egui_painter
        .draw(
            builder,
            [(size.width as f32) / sf, (size.height as f32) / sf],
            &egui_ctx,
            shapes,
        )
        .unwrap();
}
