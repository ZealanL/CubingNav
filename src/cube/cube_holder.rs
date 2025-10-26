use std::fmt::Display;
use crate::cube::{AbsCubeMove, CubeView};
use crate::cube::CubeState;
use crate::cube::CubeRelFace;
use crate::cube::TurnDir;
use crate::solve::Algorithm;

// TODO: Move out of "cube" module since we use the "solve" module here?

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CubeHolder {
    pub state: CubeState,
    pub view: CubeView,
}

impl Default for CubeHolder {
    fn default() -> Self {
        Self {
            state: CubeState::default(),
            view: CubeView::default(),
        }
    }
}

impl CubeHolder {
    pub fn turn(&mut self, rel_face: CubeRelFace, turn_dir: TurnDir) {
        let abs_face = self.view.rel_face_to_abs_color(rel_face);
        self.state = self.state.do_move(
            AbsCubeMove{ face_color: abs_face, turn_dir }
        );
    }

    pub fn do_alg(&mut self, alg: &Algorithm) {
        for mv in &alg.moves {
            self.turn(mv.rel_face, mv.turn_dir);
        }
    }
}

impl Display for CubeHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.state))
    }
}