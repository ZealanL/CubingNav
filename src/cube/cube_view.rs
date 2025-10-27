use crate::cube::CubeColor;
use crate::cube::CubeRelFace;
use crate::cube::CubeRelFace::{B, D, F, L, R, U};

// A viewing perspective of the cube for translating relative turns to absolute turns
// (i.e. allowing us to translate the move "R2" into an absolute move)
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CubeView {
    pub u_face: CubeColor,
    pub d_face: CubeColor,
    pub r_face: CubeColor,
    pub l_face: CubeColor,
    pub f_face: CubeColor,
    pub b_face: CubeColor,
}

impl Default for CubeView {
    // Matches the text display view for CubeState
    fn default() -> Self {
        use CubeColor::*;
        Self {
            u_face: White,
            d_face: Yellow,
            r_face: Orange,
            l_face: Red,
            f_face: Blue,
            b_face: Green,
        }
    }
}

impl CubeView {
    pub fn rel_face_to_abs_face(self, rel_face: CubeRelFace) -> CubeColor {
        use CubeRelFace::*;
        match rel_face {
            U => self.u_face,
            D => self.d_face,
            R => self.r_face,
            L => self.l_face,
            F => self.f_face,
            B => self.b_face,
        }
    }

    pub fn abs_face_to_rel_face(self, abs_face: CubeColor) -> CubeRelFace {
        match abs_face {
            c if c == self.u_face => U,
            c if c == self.d_face => D,
            c if c == self.r_face => R,
            c if c == self.l_face => L,
            c if c == self.f_face => F,
            c if c == self.b_face => B,
            _ => panic!(),
        }
    }

    pub fn to_rel_face_array(self) -> [CubeColor; 6] {
        [self.u_face, self.d_face, self.r_face, self.l_face, self.f_face, self.b_face]
    }

    pub fn from_rel_face_array(face_array: [CubeColor; 6]) -> Self {
        CubeView {
            u_face: face_array[0],
            d_face: face_array[1],
            r_face: face_array[2],
            l_face: face_array[3],
            f_face: face_array[4],
            b_face: face_array[5],
        }
    }

    // Return this view rotated 90 degrees clockwise along this face
    pub fn rotate_along_face(self, rel_face: CubeRelFace) -> CubeView {
        use CubeRelFace::*;

        let rel_face_map = match rel_face {
            U => [U, D, B, F, R, L],
            D => [U, D, F, B, L, R],
            R => [F, B, R, L, D, U],
            L => [B, F, R, L, U, D],
            F => [L, R, U, D, F, B],
            B => [R, L, D, U, F, B],
        };

        let face_array = self.to_rel_face_array();
        let new_face_array = rel_face_map.map(
            |face| face_array[face as usize]
        );
        CubeView::from_rel_face_array(new_face_array)
    }
}