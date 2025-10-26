use std::fmt::Display;
use crate::cube::{CubeColor, CubeRelFace, TurnDir};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AbsCubeMove {
    pub face_color: CubeColor,
    pub turn_dir: TurnDir
}

impl AbsCubeMove {
    pub const SINGLE_MOVES: [AbsCubeMove; 12] = [
        AbsCubeMove{ face_color: CubeColor::White, turn_dir: TurnDir::Clockwise },
        AbsCubeMove{ face_color: CubeColor::White, turn_dir: TurnDir::CounterClockwise },
        AbsCubeMove{ face_color: CubeColor::Red, turn_dir: TurnDir::Clockwise },
        AbsCubeMove{ face_color: CubeColor::Red, turn_dir: TurnDir::CounterClockwise },
        AbsCubeMove{ face_color: CubeColor::Blue, turn_dir: TurnDir::Clockwise },
        AbsCubeMove{ face_color: CubeColor::Blue, turn_dir: TurnDir::CounterClockwise },
        AbsCubeMove{ face_color: CubeColor::Green, turn_dir: TurnDir::Clockwise },
        AbsCubeMove{ face_color: CubeColor::Green, turn_dir: TurnDir::CounterClockwise },
        AbsCubeMove{ face_color: CubeColor::Orange, turn_dir: TurnDir::Clockwise },
        AbsCubeMove{ face_color: CubeColor::Orange, turn_dir: TurnDir::CounterClockwise },
        AbsCubeMove{ face_color: CubeColor::Yellow, turn_dir: TurnDir::Clockwise },
        AbsCubeMove{ face_color: CubeColor::Yellow, turn_dir: TurnDir::CounterClockwise },
    ];
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