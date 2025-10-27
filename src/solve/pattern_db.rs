use std::collections::hash_map::Entry;
use std::collections::HashMap;
use colored::Color;
use strum::IntoEnumIterator;
use crate::cube::{AbsCubeMove, CubeColor, CubeState, TurnDir};

#[derive(Debug)]
pub struct PatternDB {
    pub map: HashMap<u64, usize>,
}

impl PatternDB {
    pub fn build(hash_fn: impl Fn(&CubeState) -> u64, expected_size: Option<usize>) -> Self {
        println!("Building pattern database:");
        let starting_state = CubeState::default();
        let starting_hash = hash_fn(&starting_state);

        let expected_size = expected_size.unwrap_or(10_000);
        let mut map = HashMap::with_capacity(expected_size);
        map.insert(starting_hash, 0);
        let mut states = vec![CubeState::default()];
        let mut next_states = Vec::with_capacity(expected_size);
        let mut cur_depth = 1;
        while !states.is_empty() {
            next_states.clear();

            for state in &states {
                for mv in AbsCubeMove::SINGLE_MOVES {
                    let next_state = state.do_move(mv);
                    let next_state_hash = hash_fn(&next_state);

                    let map_entry = map.entry(next_state_hash);
                    if matches!(map_entry, Entry::Occupied(_)) {
                        continue; // Already searched this state
                    }

                    map_entry.or_insert(cur_depth);
                    next_states.push(next_state);
                }
            }

            println!(" > Depth {cur_depth}, entries: {}...", map.len());
            if next_states.is_empty() {
                break;
            }

            // Swap states
            states.clear();
            states.append(&mut next_states);

            cur_depth += 1;
        }

        println!(" > Done at depth {}, found {} entries!", cur_depth, map.len());
        PatternDB { map }
    }
}