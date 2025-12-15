use std::ops::Mul;
use crate::cube::{CubeMove, TurnDir};
use crate::cube::cube_const::*;

//////////////

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CubeState {
    pub corn_perm: [u8; NUM_CORNERS],
    pub edge_perm: [u8; NUM_EDGES],
    pub corn_rot: [u8; NUM_CORNERS], // 0 if the white/yellow face is up, 1 if right, 2 if left
    pub edge_rot: [u8; NUM_EDGES], // 0 if aligned, 1 if flipped
}

impl CubeState {
    pub fn get_all_bytes(&self) -> Vec<u8> {
        self.corn_perm.into_iter()
            .chain(self.corn_rot.into_iter())
            .chain(self.edge_perm.into_iter())
            .chain(self.edge_rot.into_iter())
            .collect::<Vec<u8>>()
    }

    // Group every 8 data bytes into a u64, then hash them all with almost-FNV
    pub fn calc_hash(&self) -> u64 {
        let all_bytes = self.get_all_bytes();
        debug_assert!( // Must be divisible by 8
            !all_bytes.is_empty() && (all_bytes.len() % 8 == 0)
        );

        // Ref: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        const FNV_START_VAL: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut result = FNV_START_VAL;
        for i in (0..all_bytes.len()).step_by(8) {
            let slice = &all_bytes[i..i + 8];
            let as_u64 = u64::from_le_bytes(slice.try_into().unwrap());

            result = result.wrapping_mul(FNV_PRIME);
            result ^= as_u64;
        }
        result
    }

    // Apply this cube's permutations/rotations to another cube
    pub const fn apply_to(&self, other: &CubeState) -> CubeState {
        let mut results = CubeState::ZEROED_INVALID;

        const CORNER_ROT_SHIFTS: [[u8; NUM_CORNER_ROT]; NUM_CORNER_ROT] = [
            [0, 1, 2], [1, 2, 0], [2, 0, 1],
        ];

        // Corners
        let mut i = 0;
        while i < NUM_CORNERS {
            let factor_corner_idx = other.corn_perm[i] as usize;
            results.corn_perm[i] = self.corn_perm[factor_corner_idx];

            let corner_rot_shift = CORNER_ROT_SHIFTS[self.corn_rot[factor_corner_idx] as usize];
            results.corn_rot[i] = corner_rot_shift[other.corn_rot[i] as usize];
            i += 1;
        }

        // Edges
        let mut i = 0;
        while i < NUM_EDGES {
            let factor_edge_idx = other.edge_perm[i] as usize;
            results.edge_perm[i] = self.edge_perm[factor_edge_idx];
            results.edge_rot[i] = (self.edge_rot[factor_edge_idx] + other.edge_rot[i]) % (NUM_EDGE_ROT as u8);
            i += 1;
        }

        results
    }

    pub fn do_move(&self, mv: CubeMove) -> CubeState {
        let factor_turns = &Self::FACTOR_TURNS[mv.face as usize];
        use TurnDir::*;
        match mv.dir {
            Clockwise => factor_turns[0].apply_to(&self),
            CounterClockwise => factor_turns[2].apply_to(&self),
            Double => factor_turns[1].apply_to(&self),
        }
    }

    // NOTE: Unlike the reference, this does not check validity of data
    //  (e.g. repeated piece positions, out-of-bounds values, etc.)
    // Ref: https://github.com/efrantar/rob-twophase/blob/master/src/cubie.cpp#L67
    pub fn is_solvable(&self) -> bool {
        let corner_rot_sum: u8 = self.corn_rot.iter().sum();
        if corner_rot_sum % (NUM_CORNER_ROT as u8) != 0 {
            return false; // Invalid corner rot parity
        }

        let edge_rot_sum: u8 = self.edge_rot.iter().sum();
        if edge_rot_sum % (NUM_EDGE_ROT as u8) != 0 {
            return false; // Invalid edge rot parity
        }

        let fn_perm_parity_calc = |perms: &[u8]| -> bool {
            let mut parity = false;
            for i in 0..perms.len() {
                for j in 0..i {
                    if perms[j] > perms[i] {
                        parity = !parity;
                    }
                }
            }

            parity
        };

        let corner_perm_parity = fn_perm_parity_calc(&self.corn_perm);
        let edge_perm_parity = fn_perm_parity_calc(&self.edge_perm);

        // These must match
        corner_perm_parity == edge_perm_parity
    }
}

impl Default for CubeState {
    fn default() -> Self { Self::SOLVED }
}

// Shorthand for "CubeState::apply_to(&other)"
impl Mul for CubeState {
    type Output = CubeState;
    fn mul(self, other: Self) -> Self::Output {
        self.apply_to(&other)
    }
}