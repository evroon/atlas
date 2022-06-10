#![allow(dead_code)]

use crate::WinitInputHelper;
use cgmath::{InnerSpace, Matrix3, Matrix4, Point3, Rad, Vector3};
use winit::event::VirtualKeyCode;

const MOUSE_BUTTON_LEFT: usize = 0;
const MOUSE_BUTTON_RIGHT: usize = 1;
const MOUSE_BUTTON_MIDDLE: usize = 2;

pub struct Camera {
    pub position: Point3<f32>,
    pub forward: Vector3<f32>,
    pub right: Vector3<f32>,
    pub up: Vector3<f32>,

    pub aspect_ratio: f32,
    pub proj: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub world: Matrix4<f32>,
    pub world_view: Matrix4<f32>,

    pub mouse_rotation_start_coord: (f32, f32),
}

impl Camera {
    pub fn update(&mut self) {
        self.proj = cgmath::perspective(
            Rad(std::f32::consts::FRAC_PI_2),
            self.aspect_ratio,
            5.0,
            10000.0,
        );

        self.view = Matrix4::look_to_rh(self.position, self.forward, self.up);
        self.world_view = self.view * self.world;
    }
}

pub fn construct_camera() -> Camera {
    // note: In OpenGL, the origin is at the lower left
    //       In Vulkan, the origin is at the upper left,
    //       so we have to reverse the Y axis
    let forward = Vector3::new(0.0, 0.0, 1.0);
    let up = Vector3::new(0.0, 1.0, 0.0);
    Camera {
        position: Point3::new(0.0, 0.0, -3.0),
        forward,
        up,
        right: forward.cross(up),
        aspect_ratio: 1.0,
        proj: Matrix4::from_scale(1.0),
        view: Matrix4::from_scale(1.0),
        world: Matrix4::from_scale(1.0),
        world_view: Matrix4::from_scale(1.0),
        mouse_rotation_start_coord: (0.0, 0.0),
    }
}

pub trait CameraInputLogic {
    fn handle_event(&mut self, input: &WinitInputHelper);
}

impl CameraInputLogic for Camera {
    fn handle_event(&mut self, input: &WinitInputHelper) {
        let mut move_speed = 1.0; // 1 / dt
        let rotate_speed = 0.005; // rad / (px * dt)

        if input.held_shift() {
            move_speed *= 5.0;
        }

        if input.key_held(VirtualKeyCode::W) {
            self.position += self.forward * move_speed;
        }
        if input.key_held(VirtualKeyCode::S) {
            self.position -= self.forward * move_speed;
        }
        if input.key_held(VirtualKeyCode::A) {
            self.position -= self.right * move_speed;
        }
        if input.key_held(VirtualKeyCode::D) {
            self.position += self.right * move_speed;
        }
        if input.key_held(VirtualKeyCode::E) {
            self.position -= self.up * move_speed;
        }
        if input.key_held(VirtualKeyCode::F) {
            self.position += self.up * move_speed;
        }

        if input.mouse_pressed(MOUSE_BUTTON_RIGHT) {
            self.mouse_rotation_start_coord = input.mouse().unwrap_or((0.0, 0.0));
        }

        if input.mouse_held(MOUSE_BUTTON_RIGHT) {
            let diff = input.mouse_diff();
            let transform = Matrix3::from_axis_angle(self.up, Rad(-diff.0 * rotate_speed))
                * Matrix3::from_axis_angle(self.right, Rad(diff.1 * rotate_speed));

            self.forward = (transform * self.forward).normalize();

            if input.held_control() {
                self.up = (transform * self.up).normalize();
            } else {
                self.up = Vector3::unit_y();
            }

            self.right = self.forward.cross(self.up);
        }
    }
}
