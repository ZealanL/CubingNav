use rayon::iter::ParallelIterator;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use rayon::prelude::IntoParallelRefIterator;
use crate::cube::{CubeMove, CubeState};

#[derive(Debug)]
pub struct PatternDB {
    pub map: HashMap<u64, u8>,
}

impl PatternDB {
    pub fn build(hash_fn: impl Fn(&CubeState) -> u64 + Sync, expected_size: Option<usize>, max_depth: Option<u8>) -> Self {
        println!("Building pattern database:");
        let starting_state = CubeState::default();
        let starting_hash = hash_fn(&starting_state);

        let expected_size = expected_size.unwrap_or(10_000);
        let mut final_map = HashMap::with_capacity(expected_size);
        final_map.insert(starting_hash, 0);

        let mut states = vec![CubeState::default()];
        let mut cur_depth: u8 = 1;
        while !states.is_empty() && cur_depth <= max_depth.unwrap_or(u8::MAX) {
            let entries_itr_result = states.par_iter() // Use par_iter() for parallel processing
                .flat_map(|state| {
                    // Here we use a map to prevent duplicate states of the same "hash_fn()" hash
                    let mut inner_next = HashMap::with_capacity(CubeMove::ALL_TURNS.len());

                    for mv in CubeMove::ALL_TURNS {
                        let next_state = state.do_move(mv);
                        let next_state_hash = hash_fn(&next_state);
                        inner_next.insert(next_state_hash, next_state);
                    }
                    inner_next
                });

            let next_states_map: HashMap<u64, CubeState> = entries_itr_result.collect();

            // We will progress "states" to the next states
            states.clear();
            states.reserve(next_states_map.len());
            for (next_state_hash, next_state) in next_states_map {
                // This part must remain sequential to safely modify the shared 'map'
                let map_entry = final_map.entry(next_state_hash);

                // Check if the entry is Occupied (already visited at a shallower depth)
                if matches!(map_entry, Entry::Occupied(_)) {
                    continue;
                }

                // Insert the new depth and collect the state for the next iteration
                map_entry.or_insert(cur_depth);
                states.push(next_state);
            }

            println!(" > Depth {cur_depth}, entries: {}...", final_map.len());
            if states.is_empty() {
                break;
            }

            cur_depth += 1;
        }

        println!(" > Done at depth {}, found {} entries!", cur_depth, final_map.len());
        PatternDB { map: final_map }
    }
}