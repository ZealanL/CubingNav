use crate::cube::*;

// A file for constant cube states

impl CubeState {
    pub const ZEROED_INVALID: CubeState = CubeState {
        corn_perm: [0; NUM_CORNERS],
        edge_perm: [0; NUM_EDGES],
        corn_rot: [0; NUM_CORNERS],
        edge_rot: [0; NUM_EDGES],
    };

    pub const SOLVED: CubeState = CubeState {
        corn_perm: [0,1,2,3,4,5,6,7],
        edge_perm: [0,1,2,3,4,5,6,7,8,9,10,11],
        corn_rot: [0; NUM_CORNERS],
        edge_rot: [0; NUM_EDGES],
    };

    // Each face's counter-clockwise turn
    pub const FACTOR_LEFT_TURNS: [CubeState; CubeFace::COUNT] = [
        // Ref: https://github.com/efrantar/rob-twophase/blob/master/src/move.cpp
        CubeState { // U
            corn_perm: [3, 0, 1, 2, 4, 5, 6, 7], edge_perm: [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
            corn_rot:  [0; NUM_CORNERS],         edge_rot:  [0; NUM_EDGES],
        },
        CubeState { // R
            corn_perm: [1, 5, 2, 3, 0, 4, 6, 7], edge_perm: [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
            corn_rot:  [1, 2, 0, 0, 2, 1, 0, 0], edge_rot:  [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        },
        CubeState { // F
            corn_perm: [0, 2, 6, 3, 4, 1, 5, 7], edge_perm: [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
            corn_rot:  [0, 1, 2, 0, 0, 2, 1, 0], edge_rot:  [0; NUM_EDGES],
        },
        CubeState { // D
            corn_perm: [4, 1, 2, 0, 7, 5, 6, 3], edge_perm: [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
            corn_rot:  [2, 0, 0, 1, 1, 0, 0, 2], edge_rot:  [0; NUM_EDGES],
        },
        CubeState { // L
            corn_perm: [0, 1, 2, 3, 5, 6, 7, 4], edge_perm: [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
            corn_rot:  [0; NUM_CORNERS],         edge_rot: [0; NUM_EDGES],
        },
        CubeState { // B
            corn_perm: [0, 1, 3, 7, 4, 5, 2, 6], edge_perm: [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
            corn_rot:  [0, 0, 1, 2, 0, 0, 2, 1], edge_rot:  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        },
    ];

    // Second index is clockwise turn count
    pub const FACTOR_TURNS: [[CubeState; 3]; CubeFace::COUNT] = {
        let mut result = [[CubeState::ZEROED_INVALID; 3]; CubeFace::COUNT];

        let mut face_idx = 0;
        while face_idx < CubeFace::COUNT {
            let mut cur_cube = CubeState::SOLVED;

            // Rotate from 1-3 times
            let mut rot_num = 0;
            while rot_num < 3 {
                cur_cube = Self::FACTOR_LEFT_TURNS[face_idx].apply_to(&cur_cube);
                let inv_idx = 2 - rot_num; // Since we are turning left, flip it
                result[face_idx][inv_idx] = cur_cube;
                rot_num += 1;
            }

            face_idx += 1;
        }

        result
    };

    // TODO: Use
    pub const FACTOR_REFLECT_FB: CubeState = CubeState {
        corn_perm: [2, 3, 0, 1, 6, 7, 4, 5], edge_perm: [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9],
        corn_rot: [1, 2, 2, 1, 1, 2, 2, 1], edge_rot: [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    };
}