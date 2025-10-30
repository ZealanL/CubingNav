use crate::cube::{CubeState, NUM_CORNERS, NUM_EDGES, NUM_SYM_ROTS, NUM_SYM_STATES};

// Ref: https://github.com/efrantar/rob-twophase/blob/master/src/sym.cpp
// Up-down axis 90-degree rotation (4-cycle)
const ROT_90_UD: CubeState = CubeState {
    corn_perm: [1, 2, 3, 0, 5, 6, 7, 4],
    edge_perm: [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8],
    corn_rot:  [0; NUM_CORNERS],
    edge_rot:  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
};

// Custom-made rotation!
// Right-left axis 90-degree rotation (4-cycle)
const ROT_90_RL: CubeState = CubeState {
    corn_perm: [4, 5, 1, 0, 7, 6, 2, 3],
    edge_perm: [8, 5, 9, 1, 11, 7, 10, 3, 4, 6, 2, 0],
    corn_rot:  [2, 1, 2, 1, 1, 2, 1, 2],
    edge_rot:  [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
};

pub fn rot_90_ud(state: CubeState) -> CubeState {
    ROT_90_UD * (state) * (ROT_90_UD * ROT_90_UD * ROT_90_UD)
}

pub fn rot_90_rl(state: CubeState) -> CubeState {
    ROT_90_RL * state * (ROT_90_RL * ROT_90_RL * ROT_90_RL)
}

#[derive(Debug, Clone, Copy)]
struct CubeSymRot {
    lhs: CubeState,
    rhs: CubeState,
}

impl CubeSymRot {
    // This one doesn't do anything
    pub const fn empty() -> CubeSymRot {
        CubeSymRot {
            lhs: CubeState::SOLVED,
            rhs: CubeState::SOLVED,
        }
    }

    pub fn apply_to(&self, state: CubeState) -> CubeState {
        self.lhs * state * self.rhs
    }
}

const SYM_ROT_FACTORS: [CubeSymRot; NUM_SYM_ROTS] = {
    let mut result = [CubeSymRot::empty(); NUM_SYM_ROTS];

    // These are all the combinations of valid rotations
    // Every 1 represents a 90-degree turn on one axis, and every 2 represents a 90-degree turn on another
    const FACTOR_STRS: [usize; NUM_SYM_ROTS] = [
        0, 1, 2, 11, 12, 21, 22,
        111, 112, 121, 122, 211, 221, 222,
        1112, 1121, 1122, 1211, 1222, 2111, 2221,
        11121, 12111, 12221,
    ];

    let mut i = 0;
    while i < NUM_SYM_ROTS {
        let mut cur_factor = CubeState::SOLVED;

        // Walk though each digit of the encoded 1&2 integers, and apply those rotations
        let mut cur_val = FACTOR_STRS[i];
        while cur_val > 0 {
            let next_digit = cur_val % 10;

            match next_digit {
                1 => {
                    cur_factor = ROT_90_UD.apply_to(&cur_factor);
                },
                2 => {
                    cur_factor = ROT_90_RL.apply_to(&cur_factor);
                },
                _ => unimplemented!()
            }

            cur_val /= 10; // Go to next digit
        }

        result[i] = CubeSymRot {
            lhs: cur_factor,

            // After we do (factor * state) we get an invalid cube that has been rotated how we want
            // We must fix the invalid state by "competing" the rotation
            // To do this, we multiply by the same factor on the rhs until the rotation reaches 360 degrees
            // Since we are using 90-degree rotations, we gotta multiply 3 times afterward
            rhs: cur_factor.apply_to(&cur_factor).apply_to(&cur_factor), // This factor undoes the invalid rotation

            // TODO: Replace these with handmade operations? That would surely be faster...?
            //  We would need one operation per 24 rotations, or to guarantee the compiler will optimize into that...
        };
        i += 1;
    }

    result
};

impl CubeState {
    // Mirrors the cube along the R/L axis
    pub fn mirror_rl(&self) -> CubeState {
        // NOTE: You can also achieve this by multiplying and then swapping 1 and 2 corner rots
        //  However, this is faster

        let mut result = CubeState::ZEROED_INVALID;

        // TODO: Can we not use CORNER/EDGE_MAP_IDC twice for the permutations?

        for i in 0..NUM_CORNERS {
            const CORNER_MAP_IDC: [usize; NUM_CORNERS] = [1, 0, 3, 2, 5, 4, 7, 6];
            result.corn_perm[i] = CORNER_MAP_IDC[self.corn_perm[CORNER_MAP_IDC[i]] as usize] as u8;

            // Because we mirrored, clockwise becomes counter-clockwise, and vice versa
            const CORNER_ROT_MIRRORS: [u8; 3] = [0, 2, 1];
            result.corn_rot[i] = CORNER_ROT_MIRRORS[self.corn_rot[CORNER_MAP_IDC[i]] as usize];
        }

        for i in 0..NUM_EDGES {
            const EDGE_MAP_IDC: [usize; NUM_EDGES] = [2, 1, 0, 3, 6, 5, 4, 7, 9, 8, 11, 10];
            result.edge_perm[i] = EDGE_MAP_IDC[self.edge_perm[EDGE_MAP_IDC[i]] as usize] as u8;

            // Just directly map like it were rotation
            result.edge_rot[i] = self.edge_rot[EDGE_MAP_IDC[i]];
        }

        result
    }

    // Get all symmetrical states (including this one
    pub fn get_all_syms(&self) -> [CubeState; NUM_SYM_STATES] {
        let mut result = [CubeState::ZEROED_INVALID; NUM_SYM_STATES];

        for i in 0..NUM_SYM_STATES {
            let sym_rot_idx = i / 2;
            let is_mirrored = (i % 2) != 0;

            let rotated = SYM_ROT_FACTORS[sym_rot_idx].apply_to(*self);

            if is_mirrored {
                result[i] = rotated.mirror_rl();
            } else {
                result[i] = rotated;
            }
        }

        result
    }

    // Gets the symmetrically-ambiguous hash
    // Any symmetry of a cube will have the same get_sym_hash()
    // NOTE: This is somewhat expensive
    pub fn calc_sym_hash(&self) -> u64 {
        let syms = self.get_all_syms();
        syms.iter()
            .map(|sym| sym.calc_hash())
            .min()
            .unwrap()
    }
}
