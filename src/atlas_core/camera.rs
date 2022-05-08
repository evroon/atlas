use crate::WinitInputHelper;
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use winit::event::VirtualKeyCode;

pub struct Camera {
    pub position: Point3<f32>,
    pub aspect_ratio: f32,
    pub proj: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub world: Matrix4<f32>,
    pub world_view: Matrix4<f32>,
}

impl Camera {
    pub fn update(&mut self, rotation: Matrix3<f32>) {
        self.proj = cgmath::perspective(
            Rad(std::f32::consts::FRAC_PI_2),
            self.aspect_ratio,
            0.1,
            1000.0,
        );
        let scale: Matrix4<f32> = Matrix4::from_scale(1.0);

        // note: In OpenGL, the origin is at the lower left
        //       In Vulkan, the origin is at the upper left,
        //       so we have to reverse the Y axis
        self.view = Matrix4::look_to_rh(
            self.position,
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, -1.0, 0.0),
        ) * scale;

        self.world = Matrix4::from(rotation);
        self.world_view = self.view * self.world;
    }
}

pub fn construct_camera() -> Camera {
    Camera {
        position: Point3::new(0.0, 0.0, -3.0),
        aspect_ratio: 1.0,
        proj: Matrix4::from_scale(1.0),
        view: Matrix4::from_scale(1.0),
        world: Matrix4::from_scale(1.0),
        world_view: Matrix4::from_scale(1.0),
    }
}

pub trait CameraInputLogic {
    fn handle_event(&mut self, input: &WinitInputHelper);
}

impl CameraInputLogic for Camera {
    fn handle_event(&mut self, input: &WinitInputHelper) {
        let move_speed = 0.1;

        if input.key_held(VirtualKeyCode::W) {
            self.position.z += move_speed;
        }
        if input.key_held(VirtualKeyCode::S) {
            self.position.z -= move_speed;
        }
        if input.key_held(VirtualKeyCode::A) {
            self.position.x -= move_speed;
        }
        if input.key_held(VirtualKeyCode::D) {
            self.position.x += move_speed;
        }
        if input.key_held(VirtualKeyCode::E) {
            self.position.y += move_speed;
        }
        if input.key_held(VirtualKeyCode::F) {
            self.position.y -= move_speed;
        }
    }
}
