use std::ops::{Add, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec3i {
    pub x: i8,
    pub y: i8,
    pub z: i8,
}

impl Vec3i {
    pub const fn new(x: i8, y: i8, z: i8) -> Vec3i {
        Vec3i { x, y, z }
    }

    pub const fn zero() -> Vec3i {
        Vec3i::new(0, 0, 0)
    }

    pub fn manhattan_len(&self) -> i8 {
        self.x.abs() + self.y.abs() + self.z.abs()
    }

    pub fn manhattan_dist(a: &Vec3i, b: &Vec3i) -> i8 {
        (*a - *b).manhattan_len()
    }
}

impl Add for Vec3i {
    type Output = Vec3i;
    fn add(self, other: Vec3i) -> Vec3i {
        Vec3i::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3i {
    type Output = Vec3i;
    fn sub(self, other: Vec3i) -> Vec3i {
        Vec3i::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}