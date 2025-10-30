pub const NUM_CORNERS: usize = 8;
pub const NUM_EDGES: usize = 12;
pub const NUM_CORNER_ROT: usize = 3;
pub const NUM_EDGE_ROT: usize = 2;

pub const NUM_FACES: usize = 6;
pub const NUM_FACELETS_PER_FACE: usize = 9;
pub const NUM_FACELETS: usize = NUM_FACELETS_PER_FACE * NUM_FACES;

// All the possible cube multiplication factors for 90-degree rotations
// Applying all of these to the cube allows you to achieve all valid rotations
// There are 24 because you can pick any of the 6 sides as U, and any of the adjacent 4 sides as F
pub const NUM_SYM_ROTS: usize = 24;

// The total symmetrical states for a given cube
// Includes all rotations, plus all mirrors for those rotations
pub const NUM_SYM_STATES: usize = NUM_SYM_ROTS * 2;