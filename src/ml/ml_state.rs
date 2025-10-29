use crate::cube::{CubeMask, CubeState, NUM_CORNERS, NUM_CORNER_ROT, NUM_EDGES, NUM_EDGE_ROT};
pub const NUM_TOKENS_PER_CUBE_STATE: usize = NUM_CORNERS + NUM_EDGES;
pub const NUM_TOKEN_TYPES: usize =
    NUM_CORNERS*NUM_CORNERS*NUM_CORNER_ROT +
        NUM_EDGES*NUM_EDGES*NUM_EDGE_ROT + 1 /* Masked-out token */;

pub type MLStateTokens = [u16; NUM_TOKENS_PER_CUBE_STATE];

pub fn cube_to_tokens(cube_state: &CubeState, mask: &CubeMask) -> MLStateTokens {
    let mut tokens = Vec::with_capacity(NUM_TOKENS_PER_CUBE_STATE);
    for i in 0..NUM_CORNERS {
        if mask.corners[i] {
            let corner_type = cube_state.corn_perm[i];
            let corner_rot = cube_state.corn_rot[i];
            let base_token_idx =
                (i * NUM_CORNERS * NUM_CORNER_ROT) + ((corner_type as usize) * NUM_CORNER_ROT) + (corner_rot as usize);
            tokens.push(1 + base_token_idx as u16);
        } else {
            tokens.push(0);
        }
    }

    for i in 0..NUM_EDGES {
        if mask.edges[i] {
            let edge_type = cube_state.edge_perm[i];
            let edge_rot = cube_state.edge_rot[i];
            let base_token_idx =
                (i * NUM_EDGES * NUM_EDGE_ROT) + ((edge_type as usize) * NUM_EDGE_ROT) + (edge_rot as usize);
            tokens.push(1 + (NUM_CORNERS*NUM_CORNERS*NUM_CORNER_ROT) as u16 + base_token_idx as u16);
        } else {
            tokens.push(0);
        }
    }

    for token in &tokens {
        assert!((*token as usize) < NUM_TOKEN_TYPES);
    }

    tokens.try_into().unwrap()
}