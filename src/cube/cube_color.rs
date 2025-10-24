#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, strum::EnumIter, strum::FromRepr)]
#[repr(usize)]
pub enum CubeColor {
    White, Red, Blue, Orange, Green, Yellow
}

impl CubeColor {
    // Only true for white and yellow faces
    pub fn is_face_vertical(self) -> bool {
        self == CubeColor::White || self == CubeColor::Yellow
    }

    // Opposite of is_face_vertical
    pub fn is_face_horizontal(self) -> bool {
        !self.is_face_vertical()
    }
}