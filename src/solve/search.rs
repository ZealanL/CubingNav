use std::collections::{HashMap};
use std::time::Instant;
use crate::cube::{CubeState, CubeMove};

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum SearchError {
    Impossible,
    ExceededLimit
}

#[derive(Debug,  Clone, PartialEq)]
pub struct SearchOutput {
    pub moves: Vec<CubeMove>,
    pub final_state: CubeState,
    pub searched_states: u64,
    pub elapsed_secs: f32
}

// IDA* implementation
pub fn find_moves(
    start_state: &CubeState,
    cost_fn: impl Fn(&CubeState) -> f32,
    max_depth: usize,
) -> Result<SearchOutput, SearchError> {
    let start_cost = cost_fn(start_state);

    if start_cost <= 0.0 {         // Already solved
        return Ok(SearchOutput {
            moves: Vec::new(),
            final_state: start_state.clone(),
            searched_states: 0,
            elapsed_secs: 0.0,
        });
    }

    let start_time = Instant::now();
    let mut searched_states: u64 = 0;
    let mut bound = start_cost;

    loop {
        let mut path = Vec::new();
        let mut visited_in_path = HashMap::new();
        visited_in_path.insert(start_state.calc_hash(), 0);

        let op_min_cost = ida_search_recursive(
            start_state, 0.0, bound, None, // Last move
            &mut path, &mut visited_in_path, &cost_fn, &mut searched_states,
        );

        if let Some(min_cost) = op_min_cost {
            if min_cost == f32::INFINITY {
                return Err(SearchError::Impossible);
            } else if path.len() >= max_depth {
                return Err(SearchError::ExceededLimit);
            }
            bound = min_cost;
        } else {
            let elapsed_secs = start_time.elapsed().as_secs_f32();
            let final_state = {
                let mut state = *start_state;
                for mv in &path {
                    state = state.do_move(*mv);
                }
                state
            };

            return Ok(SearchOutput {
                moves: path,
                final_state,
                searched_states,
                elapsed_secs,
            });
        }
    }
}

fn ida_search_recursive(
    state: &CubeState, g_score: f32, bound: f32, last_move: Option<CubeMove>,
    path: &mut Vec<CubeMove>, visited_in_path: &mut HashMap<u64, usize>,
    cost_fn: &impl Fn(&CubeState) -> f32, searched_states: &mut u64,
) -> Option<f32> {
    *searched_states += 1;

    let h_cost = cost_fn(state);
    let f_score = g_score + h_cost;

    if f_score > bound {
        return Some(f_score);
    } else if h_cost <= 0.0 {
        // We did it! Yayyyy
        return None;
    }

    let mut min_cost = f32::INFINITY;

    for mv in CubeMove::ALL_TURNS {
        // Prune any branch where we move the same face twice in a row
        if let Some(last_mv) = last_move {
            if mv.face == last_mv.face {
                continue;
            }
        }

        let neighbor_state = state.do_move(mv);
        let neighbor_hash = neighbor_state.calc_hash();

        // Check for cycles in current path
        if let Some(&prev_depth) = visited_in_path.get(&neighbor_hash) {
            if prev_depth < path.len() {
                // We've seen this state earlier in the path, skip it
                continue;
            }
        }

        path.push(mv);
        visited_in_path.insert(neighbor_hash, path.len());

        let op_min_cost = ida_search_recursive(
            &neighbor_state, g_score + 1.0, bound, Some(mv),
            path, visited_in_path, cost_fn, searched_states,
        );

        if let Some(next_min_cost) = op_min_cost {
            min_cost = min_cost.min(next_min_cost);
        } else {
            return None; // Found (climb back up search tree)
        }

        path.pop();
        visited_in_path.remove(&neighbor_hash);
    }

    Some(min_cost)
}