use crate::cube::NUM_FACES;
// A face on a cube relative to a perspective

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, strum::Display)]
pub enum CubeFace {
    U, R, F,
    D, L, B,
}

use CubeFace::*;

impl CubeFace {
    pub const COUNT: usize = NUM_FACES;
    pub const ALL: [CubeFace; Self::COUNT] = [U, R, F, D, L, B];
    pub const CHARS: [char; Self::COUNT] = ['U', 'R', 'F', 'D', 'L', 'B'];

    pub const fn from(idx: usize)-> CubeFace {
        Self::ALL[idx]
    }

    pub const fn to_char(self) -> char {
        CubeFace::CHARS[self as usize]
    }

    pub fn from_char(c: char) -> Option<CubeFace> {
        for face in Self::ALL {
            if c == CubeFace::CHARS[face as usize] {
                return Some(face);
            }
        }
        None
    }
}