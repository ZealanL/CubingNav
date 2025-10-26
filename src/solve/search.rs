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
pub struct SearchOutput {
    pub moves: Vec<AbsCubeMove>,
    pub final_state: CubeState,
    pub searched_states: u64,
    pub elapsed_secs: f32
}

type NodeParentInfo = Option<(usize, AbsCubeMove)>;

#[derive(Debug, Copy, Clone)]
struct SearchNode {
    state: CubeState,
    g_score: f32,
    f_score: f32,
    h_cost: f32,
    parent_info: NodeParentInfo
}

impl SearchNode {
    fn new(state: CubeState, h_cost: f32, parent_info: NodeParentInfo) -> Self {
        Self {
            state,
            g_score: f32::MAX,
            f_score: f32::MAX,
            h_cost,
            parent_info
        }
    }
}



// Will search until it finds cost=0 or reaches max_moves
// Note: Requires that cost_fn is ADMISSIBLE, meaning it never overestimates the true remaining moves
pub fn find_moves_astar(start_state: &CubeState, cost_fn: impl Fn(&CubeState) -> f32, max_explored_states: usize)
                        -> Result<SearchOutput, SearchError> {

    // Ref: https://en.wikipedia.org/wiki/A*_search_algorithm

    let start_hash = start_state.calc_hash();
    let start_cost = cost_fn(start_state);
    if start_cost <= 0.0 {
        // Already solved
        return Ok(SearchOutput {
            moves: Vec::new(),
            final_state: start_state.clone(),
            searched_states: 0,
            elapsed_secs: 0.0
        });
    }

    let mut nodes: Vec<SearchNode> = vec![
        SearchNode {
            state: *start_state,
            g_score: 0.0,
            f_score: start_cost,
            h_cost: start_cost,
            parent_info: None
        }
    ];
    let mut hash_to_idx_map: HashMap<u64, usize> = HashMap::from([(start_hash, 0)]);

    type OpenSetEntry = (usize, f32);
    let mut open_set = BinaryHeap::new_by(
        |entry_a: &OpenSetEntry, entry_b: &OpenSetEntry| {
            entry_b.1.partial_cmp(&entry_a.1).unwrap()
        }
    );
    open_set.push((0, start_cost));


    let start_time = Instant::now();
    loop {
        if let Some((cur_idx, queued_f_score)) = open_set.pop() {
            let cur_node = nodes[cur_idx];

            if queued_f_score > cur_node.f_score {
                // Stale f-score entry
                continue;
            }

            for mv in AbsCubeMove::SINGLE_MOVES {

                if let Some((_, cur_node_parent_move)) = cur_node.parent_info {
                    if mv.face_color == cur_node_parent_move.face_color {
                        if mv.turn_dir != cur_node_parent_move.turn_dir {
                            // This move will just undo our current node's move
                            // Searching it is useless!
                            continue;
                        }
                    }
                }

                let neighbor_state = cur_node.state.turn(mv);
                let neighbor_hash = neighbor_state.calc_hash();

                // Get or create neighbor
                let neighbor_idx;
                if let Some(existing_neighbor_idx) = hash_to_idx_map.get(&neighbor_hash) {
                    neighbor_idx = *existing_neighbor_idx;
                } else {
                    neighbor_idx = nodes.len();
                    nodes.push(
                        SearchNode::new(neighbor_state, cost_fn(&neighbor_state), Some((cur_idx, mv)))
                    );

                    hash_to_idx_map.insert(neighbor_hash, neighbor_idx);
                }
                let neighbor = &mut nodes[neighbor_idx];

                let added_g_score = cur_node.g_score + 1.0 /* added turn cost */;

                if neighbor.h_cost <= 0.0 { // PATH SOLVED!
                    let moves = {
                        let mut cur_backtrace_idx = cur_idx;
                        let mut moves = vec![mv];
                        loop {
                            if let Some((parent_idx, prev_turn)) = nodes[cur_backtrace_idx].parent_info {
                                moves.push(prev_turn);
                                cur_backtrace_idx = parent_idx;
                            } else {
                                break;
                            }
                        }
                        moves.reverse();
                        moves
                    };

                    let elapsed_secs = start_time.elapsed().as_secs_f32();

                    return Ok(SearchOutput {
                        moves,
                        final_state: neighbor_state,
                        searched_states: nodes.len() as u64,
                        elapsed_secs,
                    })
                }

                if added_g_score < neighbor.g_score {
                    // Update neighbor
                    neighbor.parent_info = Some((cur_idx, mv));
                    neighbor.g_score = added_g_score;
                    neighbor.f_score = added_g_score + neighbor.h_cost;

                    open_set.push((neighbor_idx, neighbor.f_score));
                    if open_set.len() >= max_explored_states {
                        return Err(SearchError::ExceededLimit);
                    }
                }
            }
        } else {
            // No more open nodes (states) to explore
            return Err(SearchError::Impossible);
        }
    }
}