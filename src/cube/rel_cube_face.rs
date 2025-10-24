use strum::IntoEnumIterator;

// A face on a cube relative to a perspective

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, strum::EnumIter, strum::EnumCount, strum::FromRepr)]
#[repr(u8)]
pub enum CubeRelFace {
    U, D,
    R, L,
    F, B,
}

impl CubeRelFace {
    pub const CHARS: [char; 6] = ['U', 'D', 'R', 'L', 'F', 'B'];

    pub fn to_char(self) -> char {
        CubeRelFace::CHARS[self as usize]
    }

    pub fn from_char(c: char) -> Option<CubeRelFace> {
        for val in CubeRelFace::iter() {
            if c == CubeRelFace::CHARS[val as usize] {
                return Some(val);
            }
        }
        None
    }
}