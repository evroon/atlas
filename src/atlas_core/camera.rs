use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};

pub struct Camera {
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
        let scale: Matrix4<f32> = Matrix4::from_scale(0.25);

        self.view = Matrix4::look_at_rh(
            Point3::new(0.3, 0.3, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        ) * scale;

        self.world = Matrix4::from(rotation);
        self.world_view = self.view * self.world;
    }
}

pub fn construct_camera() -> Camera { 
    Camera {
        aspect_ratio: 1.0,
        proj: Matrix4::from_scale(1.0),
        view: Matrix4::from_scale(1.0),
        world: Matrix4::from_scale(1.0),
        world_view: Matrix4::from_scale(1.0),
    }
}
