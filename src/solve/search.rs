use std::collections::{HashMap};
use std::time::Instant;
use crate::cube::{CubeState, CubeMove, CubeMask};
use crate::ml::MLModel;
use crate::solve::{PatternDB, SavedPatternDB, SavedPatternDBs};

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

pub type SearchResult = Result<SearchOutput, SearchError>;

// IDA* implementation
pub fn find_moves(
    start_state: &CubeState,
    cost_fn: impl Fn(&CubeState) -> f32,
    max_depth: usize,
) -> SearchResult {
    let start_cost = cost_fn(start_state);
    if start_cost <= 0.0 { // Already solved
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
            bound = min_cost;

            if min_cost == f32::INFINITY {
                return Err(SearchError::Impossible);
            } else if path.len() >= max_depth {
                return Err(SearchError::ExceededLimit);
            }
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

////////////////////////////

struct BeamSearchNode {
    state: CubeState,
    total_value: f32,
    moves: Vec<CubeMove>
}

pub fn try_pdb_solve(start_state: &CubeState, solve_pattern_db: &SavedPatternDB) -> Option<Vec<CubeMove>> {
    let (pdb_found, pdb_depth) = solve_pattern_db.get_solution_depth(&start_state);
    if pdb_found {
        let mut cur_state = *start_state;
        let mut cur_moves = Vec::new();
        let mut cur_pdb_depth = pdb_depth;
        'outer: for _ in 0..pdb_depth {
            for mv in CubeMove::ALL_TURNS {
                let next_state = cur_state.do_move(mv);

                let (pdb_found, next_pdb_depth) = solve_pattern_db.get_solution_depth(&next_state);

                if pdb_found && next_pdb_depth < cur_pdb_depth {
                    cur_moves.push(mv);

                    if next_state == CubeState::SOLVED {
                        return Some(cur_moves);
                    }

                    cur_state = next_state;
                    cur_pdb_depth = next_pdb_depth;
                    continue 'outer;
                }
            }
        }
        panic!("PDB failed to solve the cube from depth {pdb_depth}, cur cube: {}", cur_state);
    } else {
        None
    }
}

pub fn beam_search(start_state: &CubeState, beam_width: usize, max_itrs: usize, nn: &MLModel, solve_pattern_db: &SavedPatternDB) -> SearchResult {
    let mut cur_nodes: Vec<BeamSearchNode> = vec![
        BeamSearchNode {
            state: *start_state,
            total_value: 0.0,
            moves: Vec::new(),
        },
    ];
    let mut states_explored = 0;

    for _ in 0..max_itrs {
        // Generate next nodes
        let mut next_nodes = Vec::with_capacity(cur_nodes.len());
        let mut all_node_states = Vec::new();
        for node in &cur_nodes {
            all_node_states.push(node.state);
        }
        let all_model_outputs = nn.infer(&all_node_states);

        for (node_idx, node) in cur_nodes.iter().enumerate() {
            let model_output = &all_model_outputs[node_idx];

            for (move_idx, mv) in CubeMove::ALL_TURNS.iter().enumerate() {
                let move_prob = model_output.move_probs[move_idx];

                let next_state = node.state.do_move(mv.opposite());
                states_explored += 1;

                let mut next_node_moves = node.moves.clone();
                next_node_moves.push(mv.opposite());

                if let Some(solve_moves) = try_pdb_solve(&next_state, solve_pattern_db) {
                    // Solution found
                    let mut moves = next_node_moves;
                    for solve_move in solve_moves { moves.push(solve_move); }
                    return Ok(SearchOutput {
                        moves,
                        final_state: next_state,
                        searched_states: states_explored,
                        elapsed_secs: 0.0,
                    });
                }

                let next_node = BeamSearchNode {
                    state: next_state,
                    total_value: node.total_value + move_prob.ln(),
                    moves: next_node_moves
                };
                next_nodes.push(next_node);
            }
        }

        // Sort next nodes and purge if it exceeds width
        if next_nodes.len() > beam_width {
            next_nodes.sort_by(|a, b| b.total_value.partial_cmp(&a.total_value).unwrap());
            next_nodes.truncate(beam_width);
        }

        cur_nodes.clear();
        cur_nodes.append(&mut next_nodes);
    }
    Err(SearchError::ExceededLimit)
}