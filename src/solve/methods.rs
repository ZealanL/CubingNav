use crate::cube::{CubeColor, CubeState, CORNER_LOCATIONS, EDGE_LOCATIONS, FACE_EDGE_INDICES, NUM_EDGES};
use crate::util::Vec3i;

fn calc_corner_dist(from_idx: usize, from_rot: u8, to_idx: usize, to_rot: u8) -> f32 {
    let location_dist = Vec3i::manhattan_dist(
        &CORNER_LOCATIONS[from_idx], &CORNER_LOCATIONS[to_idx]
    ) / 2;
    let wrong_rot = from_rot != to_rot;
    if location_dist > 0 {
        if location_dist > 1 {
            // We are guaranteed to be able to reach it with the correct rotation
            // This is because there are 3+ ways to get it there in location_dist turns,
            //  with different rotations.
            location_dist as f32
        } else {
            if wrong_rot {
                // We're 1 away but need to rotate the corner too
                // Depending on the to-from, this may be free or require another move
                1.5
            } else {
                1.0
            }
        }
    } else {
        if wrong_rot {
            3.0 // Always takes two moves to rotate a corner and reset it to the original pos
        } else {
            0.0 // Already solved
        }
    }
}

fn calc_edge_dist(from_idx: usize, from_rot: u8, to_idx: usize, to_rot: u8) -> f32 {
    let location_dist = Vec3i::manhattan_dist(
        &EDGE_LOCATIONS[from_idx], &EDGE_LOCATIONS[to_idx]
    ) / 2;
    let wrong_rot = from_rot != to_rot;
    location_dist as f32 + if wrong_rot { 2.0 } else { 0.0 }
}

pub fn h_white_cross(cube_state: &CubeState) -> f32 {
    let mut max_dist = 0.0;
    let mut total_dist = 0.0;
    for edge_idx in 0..4 {
        let target_edge_type = edge_idx as u8;
        let from_edge_idx = cube_state.find_edge_idx(target_edge_type);
        let edge_dist = calc_edge_dist(
            from_edge_idx, cube_state.edge_rots[from_edge_idx],
            edge_idx, 0,
        );
        max_dist = f32::max(max_dist, edge_dist);
        total_dist += edge_dist;
    }

    // TODO: This function is *not* perfect and is sometimes non-admissible
    max_dist * 0.3 + total_dist * 0.7
}

pub fn h_f2l_pair(cube_state: &CubeState, pair_idx: usize) -> f32 {
    assert!(pair_idx < 4);
    let corner_idx = pair_idx;
    let edge_idx = pair_idx + 4;

    let from_corner_idx = cube_state.find_corner_idx(corner_idx as u8);
    let from_edge_idx = cube_state.find_edge_idx(edge_idx as u8);
    let corner_dist = calc_corner_dist(from_corner_idx, cube_state.corner_rots[from_corner_idx], corner_idx, 0);
    let edge_dist = calc_edge_dist(from_edge_idx, cube_state.edge_rots[from_edge_idx], edge_idx, 0);

    (h_white_cross(&cube_state) + corner_dist + edge_dist) * 0.5
}