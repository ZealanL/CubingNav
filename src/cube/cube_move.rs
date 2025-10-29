use crate::cube::CubeFace;
use crate::cube::CubeFace::*;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum TurnDir {
    Clockwise, // Clockwise
    CounterClockwise, // Counter-Clockwise
    Double,
}

impl TurnDir {
    pub const COUNT: usize = 3;
    pub const ALL: [TurnDir; Self::COUNT] = [TurnDir::Clockwise, TurnDir::CounterClockwise, TurnDir::Double];

    pub const fn opposite(&self) -> TurnDir {
        match *self {
            TurnDir::Clockwise => TurnDir::CounterClockwise,
            TurnDir::CounterClockwise => TurnDir::Clockwise,
            TurnDir::Double => TurnDir::Double,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CubeMove {
    pub face: CubeFace,
    pub dir: TurnDir
}

impl CubeMove {
    pub const ALL_TURNS: [CubeMove; CubeFace::COUNT * TurnDir::COUNT] = {
        let mut output = [CubeMove{ face: U, dir: TurnDir::Clockwise }; CubeFace::COUNT * TurnDir::COUNT];
        let mut i = 0;
        while i < CubeFace::COUNT {
            let face = CubeFace::ALL[i];
            let mut j = 0;
            while j < TurnDir::COUNT {
                output[i * TurnDir::COUNT + j] = CubeMove{ face, dir: TurnDir::ALL[j] };
                j += 1;
            }
            i += 1;
        }
        output
    };

    pub const fn opposite(&self) -> CubeMove {
        CubeMove {
            face: self.face,
            dir: self.dir.opposite()
        }
    }
}