use crate::cube::{RelCubeMove, CubeRelFace};
use crate::cube::TurnDir;

#[derive(Debug, Clone, PartialEq)]
pub struct Algorithm {
    pub moves: Vec<RelCubeMove>,
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
                if let Some(rel_face) = CubeRelFace::from_char(c) {
                    alg.moves.push(
                        RelCubeMove {
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
                    if last_move.turn_dir == TurnDir::Clockwise {
                        last_move.turn_dir = prev_turn_dir;
                    } else {
                        return Err("Found multiple post-letter characters".to_string());
                    }
                } else {
                    return Err(format!("Found post-letter character '{c}' before any letter"));
                }
            }
        }

        Ok(alg)
    }
}