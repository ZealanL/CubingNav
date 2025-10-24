use std::collections::{HashMap, HashSet};
use std::time::Instant;
use strum::IntoEnumIterator;
use crate::cube::{CubeColor, CubeState, TurnDir, AbsCubeMove};
use binary_heap_plus::BinaryHeap;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum SearchError {
    Impossible,
    ExceededLimit
}

#[derive(Debug,  Clone, PartialEq)]
pub struct SearchResult {
    pub moves: Vec<AbsCubeMove>,
    pub final_state: CubeState,
    pub searched_states: u64,
    pub elapsed_secs: f32
}

// Will search until it finds cost=0 or reaches max_moves
pub fn find_moves(start_state: &CubeState, cost_fn: impl Fn(&CubeState) -> f32, max_explored_states: usize)
                  -> Result<SearchResult, SearchError> {

    // Ref: https://en.wikipedia.org/wiki/A*_search_algorithm

    let start_hash = start_state.calc_hash();
    let start_cost = cost_fn(start_state);
    if start_cost <= 0.0 {
        // Already solved
        return Ok(SearchResult {
            moves: Vec::new(),
            final_state: start_state.clone(),
            searched_states: 0,
            elapsed_secs: 0.0
        });
    }

    let mut found_states: HashMap<u64, CubeState> = HashMap::from([
        (start_hash, *start_state)
    ]);

    type OpenSetEntry = (u64, f32);

    let mut open_set = BinaryHeap::new_by(
        |entry_a: &OpenSetEntry, entry_b: &OpenSetEntry| {
            entry_b.1.partial_cmp(&entry_a.1).unwrap()
        }
    );
    open_set.push((start_hash, start_cost));

    let mut parents: HashMap<u64, (u64, AbsCubeMove)> = HashMap::new();
    let mut g_scores: HashMap<u64, f32> = HashMap::from([
        (start_hash, 0.0),
    ]);
    let mut f_scores: HashMap<u64, f32> = HashMap::from([
        (start_hash, start_cost),
    ]);

    let start_time = Instant::now();

    loop {
        if let Some((best_open_state_hash, _)) = open_set.pop() {
            let cur_hash = best_open_state_hash;
            let cur_state = *found_states.get(&cur_hash).unwrap();
            let cur_g_score = *g_scores.get(&cur_hash).unwrap_or(&f32::MAX);
            for face in CubeColor::iter() {
                for turn_dir in [TurnDir::Clockwise, TurnDir::CounterClockwise] {
                    let mv = AbsCubeMove {
                        face_color: face,
                        turn_dir
                    };

                    let neighbor_state = cur_state.turn(mv);
                    let neighbor_hash = neighbor_state.calc_hash();

                    let added_g_score = cur_g_score + 1.0;

                    found_states.insert(neighbor_hash, neighbor_state);

                    let neighbor_h_cost = cost_fn(&neighbor_state);
                    let neighbor_f_score = added_g_score + neighbor_h_cost;
                    let neighbor_g_score = *g_scores.get(&neighbor_hash).unwrap_or(&f32::MAX);

                    if neighbor_h_cost <= 0.0 { // Path solved
                        let moves = {
                            let mut cur_backtrace_hash = cur_hash;
                            let mut moves = vec![mv];
                            loop {
                                if let Some((prev_hash, prev_turn)) = parents.get(&cur_backtrace_hash) {
                                    moves.push(*prev_turn);
                                    cur_backtrace_hash = *prev_hash;
                                } else {
                                    break;
                                }
                            }
                            moves.reverse();
                            moves
                        };

                        let elapsed_secs = start_time.elapsed().as_secs_f32();

                        return Ok(SearchResult {
                            moves,
                            final_state: neighbor_state,
                            searched_states: found_states.len() as u64,
                            elapsed_secs,
                        })
                    }

                    if added_g_score < neighbor_g_score {
                        parents.insert(neighbor_hash, (cur_hash, mv));
                        g_scores.insert(neighbor_hash, added_g_score);
                        f_scores.insert(neighbor_hash, neighbor_f_score);

                        open_set.push((neighbor_hash, neighbor_f_score));
                        if open_set.len() >= max_explored_states {
                            return Err(SearchError::ExceededLimit);
                        }
                    }
                }
            }
        } else {
            // No more open nodes (states) to explore
            return Err(SearchError::Impossible);
        }
    }
}