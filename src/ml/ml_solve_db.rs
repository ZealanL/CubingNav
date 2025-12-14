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
    last_moves: Vec<u8>
}

impl Default for MLSolveDB {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            last_moves: Vec::new()
        }
    }
}

impl MLSolveDB {
    pub fn len(&self) -> usize {
        self.states.len()
    }

    // NOTE: moves_frac_exp is the scaling/"branching" factor that makes scrambles with more moves more likely
    pub fn generate(num_entries: usize, min_moves: u8, max_moves: u8, moves_frac_exp: f32, mask: &CubeMask) -> MLSolveDB {
        assert!(min_moves > 0);
        assert!(max_moves > min_moves);
        println!("Generating {num_entries} database entries with max_moves={max_moves}...");

        let mut result = MLSolveDB::default();
        result.states.reserve(num_entries);
        result.last_moves.reserve(num_entries);
        for _ in 0..num_entries {
            let rng = &mut rand::rng();

            let moves_frac = rng.random_range(0.0..1.0f32).powf(1.0 / moves_frac_exp);
            let moves_range_size = max_moves - min_moves;
            let move_count = ((min_moves as f32) + ((moves_range_size as f32) * moves_frac).round()) as usize;

            let mut cur_cube = CubeState::SOLVED;
            let mut found_hashes = HashSet::from([
                cur_cube.calc_masked_hash(mask)
            ]);
            let max_retires = (max_moves as usize) * 100;
            let mut total_retries = 0;
            let mut last_move_idx = None;
            while found_hashes.len() < (move_count + 1) { // Until we've made "move_count" unique moves
                let move_idx = rng.random_range(0..CubeMove::ALL_TURNS.len());
                let mv = CubeMove::ALL_TURNS[move_idx];

                let next_cube = cur_cube.do_move(mv);
                let next_hash = next_cube.calc_masked_hash(mask);
                if found_hashes.insert(next_hash) {
                    cur_cube = next_cube;
                    last_move_idx = Some(move_idx);
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
            result.last_moves.push(last_move_idx.unwrap() as u8);
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

        let move_counts = Array1::from(self.last_moves.clone());
        ndarray_npy::write_npy(base_path.join("db_last_moves.npy"), &move_counts).unwrap();

        println!(" > Done!")
    }
}