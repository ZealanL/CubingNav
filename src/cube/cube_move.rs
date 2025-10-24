use std::fmt::Display;
use crate::cube::{CubeColor, CubeRelFace, TurnDir};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AbsCubeMove {
    pub face_color: CubeColor,
    pub turn_dir: TurnDir
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RelCubeMove {
    pub rel_face: CubeRelFace,
    pub turn_dir: TurnDir
}

impl Display for RelCubeMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let letter_char = self.rel_face.to_char();
        let suffix = match self.turn_dir {
            TurnDir::Clockwise => "",
            TurnDir::CounterClockwise => "'",
            TurnDir::Double => "2"
        };

        f.write_fmt(format_args!("{letter_char}{suffix}"))
    }
}