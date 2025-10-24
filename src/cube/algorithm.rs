use std::fmt::Display;
use crate::cube::CubeRelFace;
use crate::cube::turn_dir::TurnDir;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AlgorithmMove {
    pub rel_face: CubeRelFace,
    pub turn_dir: TurnDir
}

impl Display for AlgorithmMove {
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

#[derive(Debug, Clone, PartialEq)]
pub struct Algorithm {
    pub moves: Vec<AlgorithmMove>,
}

impl Algorithm {
    pub fn new() -> Self {
        Self { moves: Vec::new() }
    }

    // Parse an algorithm from a string like "R U R' U R U2 R"
    pub fn from_str(input: &str) -> Result<Algorithm, String> {
        let mut alg = Algorithm::new();

        for c in input.chars() {
            if c.is_whitespace() { continue; }
            if c.is_alphabetic() {
                if let Some(rel_face) = CubeRelFace::from_char(c) {
                    alg.moves.push(
                        AlgorithmMove {
                            rel_face,
                            turn_dir: TurnDir::Clockwise // May be updated later
                        }
                    )
                } else {
                    return Err(format!("Invalid cube face letter: '{c}'"));
                }
            } else {
                let prev_turn_dir = match c {
                    '\'' => {
                        // Prime (counter-clockwise) symbol
                        TurnDir::CounterClockwise
                    },
                    '2' => {
                        // Double-move symbol
                        TurnDir::Double
                    },
                    _ => {
                        return Err(format!("Invalid post-letter character: '{c}'"));
                    }
                };

                if let Some(last_move) = alg.moves.last_mut() {
                    last_move.turn_dir = prev_turn_dir;
                } else {
                    return Err(format!("Found post-letter character '{c}' before any letter"));
                }
            }
        }

        Ok(alg)
    }
}