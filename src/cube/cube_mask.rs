use crate::cube::{CubeState, NUM_CORNERS, NUM_EDGES};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CubeMask {
    pub corners: [bool; NUM_CORNERS],
    pub edges: [bool; NUM_EDGES],
}

impl CubeMask {
    pub fn none() -> Self {
        Self {
            corners: [false; NUM_CORNERS],
            edges: [false; NUM_EDGES],
        }
    }

    pub fn all() -> Self {
        Self {
            corners: [true; NUM_CORNERS],
            edges: [true; NUM_EDGES],
        }
    }

    pub fn calc_masked_hash(&self, cube_state: &CubeState) -> u64 {
        let mut masked_state = cube_state.clone();
        for i in 0..NUM_CORNERS {
            if !self.corners[i] {
                masked_state.corner_types[i] = 0;
                masked_state.corner_rots[i] = 0;
            }
        }

        for i in 0..NUM_EDGES {
            if !self.edges[i] {
                masked_state.edge_types[i] = 0;
                masked_state.edge_rots[i] = 0;
            }
        }

        masked_state.calc_hash()
    }
}
