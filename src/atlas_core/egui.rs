use std::sync::Arc;

use egui::{epaint::ClippedShape, Context, TextStyle, Ui};
use egui_vulkano::{Painter, UpdateTexturesResult};
use egui_winit::State;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::Device,
    render_pass::{RenderPass, Subpass},
    swapchain::Surface,
    sync::{self, FenceSignalFuture, GpuFuture},
};
use winit::window::Window;

use super::{
    renderer::deferred::{DebugPreviewBuffer, RendererParams},
    system::System,
};

pub enum FrameEndFuture<F: GpuFuture + 'static> {
    FenceSignalFuture(FenceSignalFuture<F>),
    BoxedFuture(Box<dyn GpuFuture>),
}

pub struct EguiData {
    pub egui_ctx: Context,
    pub egui_winit: State,
    pub egui_painter: Painter,
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

pub fn get_egui_context(system: &System, render_pass: &Arc<RenderPass>) -> EguiData {
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
        Subpass::from(render_pass.clone(), 2).expect("Could not create egui subpass"),
    )
    .expect("Could not create egui painter");

    EguiData {
        egui_ctx,
        egui_winit,
        egui_painter,
    }
}

fn preview_type_checkbox_item(
    ui: &mut Ui,
    item: DebugPreviewBuffer,
    value: &mut DebugPreviewBuffer,
) -> egui::Response {
    ui.selectable_value(value, item, item.get_text())
}

impl EguiData {
    pub fn render_egui(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        surface: &Arc<Surface<Window>>,
        shapes: Vec<ClippedShape>,
    ) {
        let size = surface.window().inner_size();
        let sf: f32 = surface.window().scale_factor() as f32;
        self.egui_painter
            .draw(
                builder,
                [(size.width as f32) / sf, (size.height as f32) / sf],
                &self.egui_ctx,
                shapes,
            )
            .unwrap();
    }

    pub fn update_textures_egui(
        &mut self,
        system: &System,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        params: &mut RendererParams,
    ) -> (Vec<ClippedShape>, bool) {
        self.egui_ctx
            .begin_frame(self.egui_winit.take_egui_input(system.surface.window()));

        egui::Window::new("Monitoring").show(&self.egui_ctx, |ui| {
            ui.label(system.info.device_name.clone());
            ui.label(system.info.device_type.clone());
            ui.label(format!(
                "delta time: {:.2} ms",
                system.performance_info.delta_time_ms
            ));
            ui.label(format!(
                "render time: {:.2} ms",
                system.performance_info.render_time_ms
            ));

            ui.label("Ambient light color");
            ui.color_edit_button_rgba_unmultiplied(&mut params.ambient_color);
            ui.end_row();

            ui.label("Directional light color");
            ui.color_edit_button_rgba_unmultiplied(&mut params.directional_color);
            ui.end_row();

            egui::ComboBox::from_label("Preview")
                .selected_text(params.preview_buffer.get_text())
                .show_ui(ui, |ui| {
                    preview_type_checkbox_item(
                        ui,
                        DebugPreviewBuffer::FinalOutput,
                        &mut params.preview_buffer,
                    );
                    preview_type_checkbox_item(
                        ui,
                        DebugPreviewBuffer::Albedo,
                        &mut params.preview_buffer,
                    );
                    preview_type_checkbox_item(
                        ui,
                        DebugPreviewBuffer::Normal,
                        &mut params.preview_buffer,
                    );
                    preview_type_checkbox_item(
                        ui,
                        DebugPreviewBuffer::Position,
                        &mut params.preview_buffer,
                    );
                });
            ui.end_row();
        });

        // Get the shapes from egui
        let egui_output = self.egui_ctx.end_frame();
        let platform_output = egui_output.platform_output;
        self.egui_winit.handle_platform_output(
            system.surface.window(),
            &self.egui_ctx,
            platform_output,
        );

        let result = self
            .egui_painter
            .update_textures(egui_output.textures_delta, builder)
            .expect("egui texture error");

        let wait_for_last_frame = result == UpdateTexturesResult::Changed;
        (egui_output.shapes, wait_for_last_frame)
    }
}
