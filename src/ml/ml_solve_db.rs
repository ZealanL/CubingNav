use std::collections::HashSet;
use std::path::Path;
use ndarray::{Array1, ArrayView};
use rand::distr::uniform::{SampleRange, SampleUniform};
use rand::Rng;
use crate::cube::*;
use crate::ml;
use crate::ml::{MLStateTokens, NUM_TOKENS_PER_CUBE_STATE};

pub struct MLSolveDB {
    states: Vec<MLStateTokens>,
    move_counts: Vec<u8>
}

impl Default for MLSolveDB {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            move_counts: Vec::new()
        }
    }
}

impl MLSolveDB {
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn generate(num_entries: usize, max_moves: u8, mask: &CubeMask) -> MLSolveDB {
        assert!(max_moves > 1);
        println!("Generating {num_entries} database entries with max_moves={max_moves}...");

        // Scaling/"branching" factor that makes positions with more moves more likely
        const MOVES_FRAC_EXP: f32 = 3.0;

        let mut result = MLSolveDB::default();
        result.states.reserve(num_entries);
        result.move_counts.reserve(num_entries);
        for _ in 0..num_entries {
            let rng = &mut rand::rng();

            let moves_frac = rng.random_range(0.0..1.0f32).powf(1.0 / MOVES_FRAC_EXP);
            let move_count = (max_moves as f32 * moves_frac).round() as usize;

            let mut cur_cube = CubeState::solved();
            let mut found_hashes = HashSet::from([
                mask.calc_masked_hash(&cur_cube),
            ]);
            let max_retires = (max_moves as usize) * 100;
            let mut total_retries = 0;
            while found_hashes.len() < move_count { // Until we've made "move_count" unique moves
                let move_idx = rng.random_range(0..AbsCubeMove::SINGLE_MOVES.len());
                let mv = AbsCubeMove::SINGLE_MOVES[move_idx];

                let next_cube = cur_cube.do_move(mv);
                let next_hash = mask.calc_masked_hash(&next_cube);
                if found_hashes.insert(next_hash) {
                    cur_cube = next_cube;
                } else {
                    // Already found, ignore and retry
                    total_retries += 1;
                    if total_retries > max_retires {
                        panic!(
                            "Reached max retries when trying to find a unique move (move number = {})",
                            found_hashes.len()
                        );
                    }
                }
            }

            let tokens = ml::cube_to_tokens(&cur_cube, mask);
            result.states.push(tokens);
            result.move_counts.push(move_count as u8);
        }

        println!(" > Done!");
        result
    }

    pub fn save_to_np_files(&self, base_path: &Path) {
        println!("Saving database (size={}) to {:?}...", self.len(), base_path);

        let _ = std::fs::create_dir_all(base_path);

        let state_tokens_flat: Vec<u16> = self.states.iter()
            .flat_map(|entry| entry.iter().cloned())
            .collect();

        let tokens = ArrayView::from_shape(
            (self.len(), NUM_TOKENS_PER_CUBE_STATE), &state_tokens_flat
        ).unwrap();
        ndarray_npy::write_npy(base_path.join("db_tokens.npy"), &tokens).unwrap();

        let move_counts = Array1::from(self.move_counts.clone());
        ndarray_npy::write_npy(base_path.join("db_move_counts.npy"), &move_counts).unwrap();

        println!(" > Done!")
    }
}