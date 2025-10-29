use crate::cube::{CubeFace, CubeMove, CubeState};
use crate::cube::TurnDir::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Algorithm {
    pub moves: Vec<CubeMove>,
}

impl Algorithm {
    pub fn new() -> Self {
        Self { moves: Vec::new() }
    }

    // Parse an algorithm from a string like "R U R' U R U2 R"
    // TODO: Support middle-slice moves as well as wide-moves (e.x. lowercase 'r')
    pub fn from_str(input: &str) -> Result<Algorithm, String> {
        let mut alg = Algorithm::new();

        for c in input.chars() {
            if c.is_whitespace() { continue; }
            if c.is_alphabetic() {
                if let Some(face) = CubeFace::from_char(c) {
                    alg.moves.push(
                        CubeMove {
                            face,
                            dir: Clockwise // May be updated later
                        }
                    );
                } else {
                    return Err(format!("Invalid cube face letter: '{c}'"));
                }
            } else {
                // Modifier for a move
                if !alg.moves.is_empty() {
                    let last_move = alg.moves.last_mut().unwrap();
                    match c {
                        '\'' => {
                            // Invert dir
                            *last_move = last_move.opposite();
                        },
                        '2' => {
                            // Make the last move's turn double
                            if last_move.dir != Double {
                                *last_move = CubeMove {
                                    face: last_move.face, dir: Double
                                };
                            } else {
                                return Err("Cannot apply double-turn modifier multiple times".to_string());
                            }
                        },
                        _ => {
                            return Err(format!("Invalid face turn modifier: '{c}'"))
                        }
                    }
                } else {
                    return Err(format!("Found post-letter character '{c}' before any letter"));
                }
            }
        }

        Ok(alg)
    }
}

impl CubeState {
    pub fn do_alg(&self, alg: &Algorithm) -> CubeState {
        let mut result = *self;
        for mv in &alg.moves {
            result = result.do_move(*mv);
        }
        result
    }
}