use crate::cube::{CubeState, NUM_CORNERS, NUM_EDGES};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CubeMask {
    pub corners: [bool; NUM_CORNERS],
    pub edges: [bool; NUM_EDGES],
}

impl CubeMask {
    pub const fn none() -> Self {
        Self {
            corners: [false; NUM_CORNERS],
            edges: [false; NUM_EDGES],
        }
    }

    pub const fn all() -> Self {
        Self {
            corners: [true; NUM_CORNERS],
            edges: [true; NUM_EDGES],
        }
    }

    pub const fn from_byte_arrays(corners_bytes: [u8; NUM_CORNERS], edges_bytes: [u8; NUM_EDGES]) -> CubeMask {
        let mut result = CubeMask::none();

        let mut i = 0;
        while i < NUM_CORNERS {
            result.corners[i] = corners_bytes[i] > 0;
            i += 1;
        }

        let mut i = 0;
        while i < NUM_EDGES {
            result.edges[i] = edges_bytes[i] > 0;
            i += 1;
        }

        result
    }

    pub fn true_count(&self) -> usize {
        let mut count = 0;
        for b in self.corners {
            if b { count += 1; }
        }
        for b in self.edges {
            if b { count += 1; }
        }
        count
    }

    pub fn is_empty(&self) -> bool {
        self == &Self::none()
    }
}

fn get_masked_state(cube_state: &CubeState, mask: &CubeMask) -> CubeState {
    let mut masked_state = *cube_state;
    for i in 0..NUM_CORNERS {
        if !mask.corners[i] {
            masked_state.corn_perm[i] = 0;
            masked_state.corn_rot[i] = 0;
        }
    }

    for i in 0..NUM_EDGES {
        if !mask.edges[i] {
            masked_state.edge_perm[i] = 0;
            masked_state.edge_rot[i] = 0;
        }
    }
    masked_state
}

impl CubeState {
    pub fn calc_masked_hash(&self, mask: &CubeMask) -> u64 {
        get_masked_state(self, mask).calc_hash()
    }

    // NOTE: We can't just compute the sym hash on the masked cube, as that will break the sym hash
    pub fn calc_masked_sym_hash(&self, mask: &CubeMask) -> u64 {
        let syms = self.get_all_syms();
        syms.iter()
            .map(|sym| sym.calc_masked_hash(mask))
            .min()
            .unwrap()
    }
}