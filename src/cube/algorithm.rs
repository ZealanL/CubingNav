use std::fmt::{Display, Formatter};
use rand::Rng;
use crate::cube::{CubeFace, CubeMove, CubeState, TurnDir};
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

    pub fn mirror_rl(&self) -> Algorithm {
        let mut result = self.clone();
        for mv in &mut result.moves {
            // Dir is always flipped
            mv.dir = mv.dir.opposite();

            // Flip R and L faces
            mv.face = match mv.face {
                CubeFace::R => { CubeFace::L },
                CubeFace::L => { CubeFace::R },
                _ => { mv.face }
            };
        }

        result
    }

    // NOTE: Avoids picking the same face twice in a row
    pub fn generate_random(length: usize) -> Algorithm {
        let mut moves: Vec<CubeMove> = Vec::new();

        for _ in 0..length {

            let face_idx = if moves.is_empty() {
                // Pick any face to start
                rand::random_range(0..CubeFace::COUNT)
            } else {
                let prev_face_idx = moves.last().unwrap().face as usize;

                // Don't pick the same face twice in a row
                // We can just pick with 1 less option at the end, then shift to match what's available
                let collapsed_idx = rand::random_range(0..(CubeFace::COUNT-1));
                if collapsed_idx < prev_face_idx {
                    collapsed_idx
                } else {
                    collapsed_idx + 1
                }
            };

            let turn_dir_idx = rand::random_range(0..TurnDir::COUNT);

            let mv = CubeMove {
                face: CubeFace::ALL[face_idx],
                dir: TurnDir::ALL[turn_dir_idx],
            };
            moves.push(mv);
        }

        Algorithm {
            moves
        }
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

impl Display for Algorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(first_move) = self.moves.first() {
            let mut output = format!("{}", first_move);
            for mv in self.moves.iter().skip(1) {
                output += &format!(" {mv}");
            }

            f.write_str(&output)
        } else {
            f.write_str("[empty Algorithm]")
        }
    }
}