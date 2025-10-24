use std::fmt::Display;
use colored::Colorize;
use strum::IntoEnumIterator;
use crate::cube::{AbsCubeMove, CubeColor};
use crate::cube::turn_dir::TurnDir;

pub const NUM_FACES_COLORS: usize = 6;
pub const NUM_CORNERS: usize = 8;
pub const NUM_EDGES: usize = 12;
pub const NUM_CORNER_ROT: usize = 3;
pub const NUM_EDGE_ROT: usize = 2;

// Order: Clockwise from the corner's normal, top/botton first
pub const CORNER_COLORS: [[CubeColor; NUM_CORNER_ROT]; NUM_CORNERS] = {
    use CubeColor::*;
    [
        [White, Orange, Blue],
        [White, Blue, Red],
        [White, Red, Green],
        [White, Green, Orange],

        [Yellow, Blue, Orange],
        [Yellow, Red, Blue],
        [Yellow, Green, Red],
        [Yellow, Orange, Green],
    ]
};

pub const EDGE_COLORS: [[CubeColor; NUM_EDGE_ROT]; NUM_EDGES] = {
    use CubeColor::*;
    [
        [White, Blue],
        [White, Red],
        [White, Green],
        [White, Orange],

        [Orange, Blue],
        [Blue, Red],
        [Red, Green],
        [Green, Orange],

        [Yellow, Blue],
        [Yellow, Red],
        [Yellow, Green],
        [Yellow, Orange],
    ]
};

// The four corner indices on each face
// Starts at top right, rotates around clockwise
pub const FACE_CORNER_INDICES: [[usize; 4]; NUM_FACES_COLORS] = [
    [0, 1, 2, 3],
    [1, 5, 6, 2],
    [0, 4, 5, 1],
    [3, 7, 4, 0],
    [2, 6, 7, 3],
    [5, 4, 7, 6]
];

// The four edge indices on each face
// Starts at top, rotates around clockwise
pub const FACE_EDGE_INDICES: [[usize; 4]; NUM_FACES_COLORS] = [
    [ 0,  1,  2,  3],
    [ 1,  5,  9,  6],
    [ 0,  4,  8,  5],
    [ 3,  7, 11,  4],
    [ 2,  6, 10,  7],
    [11, 10,  9,  8]
];

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, strum::EnumIter)]
pub enum FaceletPos {
    TopLeft,    TopMid,    TopRight,
    MidLeft,    Center,    MidRight,
    BottomLeft, BottomMid, BottomRight
}

impl FaceletPos {
    pub fn rotate_clockwise(self) -> FaceletPos {
        use FaceletPos::*;
        match self {
            // Corners
            TopLeft     => TopRight,
            TopRight    => BottomRight,
            BottomRight => BottomLeft,
            BottomLeft  => TopLeft,

            // Edges
            TopMid      => MidRight,
            MidRight    => BottomMid,
            BottomMid   => MidLeft,
            MidLeft     => TopMid,

            // Self-explanatory lol
            Center => Center
        }
    }

    pub fn rotate_counterclockwise(self) -> FaceletPos {
        self.rotate_clockwise().rotate_clockwise().rotate_clockwise()
    }
}

//////////////

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CubeState {
    pub corner_types: [u8; NUM_CORNERS],
    pub corner_rots: [u8; NUM_CORNERS], // 0 if the white/yellow face is up, 1 if right, 2 if left
    pub edge_types: [u8; NUM_EDGES],
    pub edge_rots: [u8; NUM_EDGES],
}

impl CubeState {
    pub const fn solved() -> Self {
        Self {
            corner_types: [0,1,2,3,4,5,6,7],
            corner_rots: [0; NUM_CORNERS],
            edge_types: [0,1,2,3,4,5,6,7,8,9,10,11],
            edge_rots: [0; NUM_EDGES],
        }
    }

    pub fn corner_equals(&self, other: &CubeState, corner_idx: usize) -> bool {
        (self.corner_types[corner_idx] == other.corner_types[corner_idx])
            && (self.corner_rots[corner_idx] == other.corner_rots[corner_idx])
    }

    pub fn edge_equals(&self, other: &CubeState, edge_idx: usize) -> bool {
        (self.edge_types[edge_idx] == other.edge_types[edge_idx])
            && (self.edge_rots[edge_idx] == CubeState::solved().edge_rots[edge_idx])
    }

    fn make_turn_from_indices(&self,
                    corner_indices: [usize; 4], edge_indices: [usize; 4],
                    corner_twists: [u8; 4], edge_flips: [u8; 4]
    ) -> CubeState {
        let mut new_corner_types = self.corner_types.clone();
        let mut new_edge_types = self.edge_types.clone();
        let mut new_corner_rots = self.corner_rots.clone();
        let mut new_edge_rots = self.edge_rots.clone();
        for i in 0..4 {
            let shift_i = (i+1) % 4;
            let o_corner_idx = corner_indices[i];
            let n_corner_idx = corner_indices[shift_i];
            let o_edge_idx = edge_indices[i];
            let n_edge_idx = edge_indices[shift_i];
            { // Update pos
                new_corner_types[n_corner_idx] = self.corner_types[o_corner_idx];
                new_edge_types  [n_edge_idx] = self.edge_types[o_edge_idx];
            }

            { // Update rot
                let new_corner_rot = (self.corner_rots[o_corner_idx] + corner_twists[i]) % (NUM_CORNER_ROT as u8);
                new_corner_rots[n_corner_idx] = new_corner_rot;

                let new_edge_rot = self.edge_rots[o_edge_idx] ^ edge_flips[i];
                new_edge_rots[n_edge_idx] = new_edge_rot;
            }
        }

        CubeState {
            corner_types: new_corner_types,
            corner_rots: new_corner_rots,
            edge_types: new_edge_types,
            edge_rots: new_edge_rots,
        }
    }

    fn clockwise_turn(&self, face: CubeColor) -> CubeState {
        // TODO: These are constants, move them elsewhere
        let corner_twists = if face.is_face_horizontal() { [1, 2, 1, 2] } else { [0,0,0,0] };
        let edge_flips = if face.is_face_horizontal() { [0, 0, 1, 1] } else { [0,0,0,0] };

        self.make_turn_from_indices(
            FACE_CORNER_INDICES[face as usize], FACE_EDGE_INDICES[face as usize],
            corner_twists, edge_flips
        )
    }

    pub fn turn(&self, abs_move: AbsCubeMove) -> CubeState {
        // TODO: Make better method
        match abs_move.turn_dir {
            TurnDir::Clockwise => self.clockwise_turn(abs_move.face_color),
            TurnDir::Double => self
                .clockwise_turn(abs_move.face_color)
                .clockwise_turn(abs_move.face_color),
            TurnDir::CounterClockwise => self
                .clockwise_turn(abs_move.face_color)
                .clockwise_turn(abs_move.face_color)
                .clockwise_turn(abs_move.face_color),
        }
    }

    // Given a corner index, returns that corners color, with the color index shifted up by "rotation_shift"
    // E.x. To get the upwards-facing color of the first corner on the white/up face, call with (0, 0)
    fn get_corner_face_color(&self, corner_idx: usize, rotation_shift: u8) -> CubeColor {
        let corner_type = self.corner_types[corner_idx];
        let corner_rot = self.corner_rots[corner_idx];
        
        let rel_corner_rot = (corner_rot + rotation_shift) % (NUM_CORNER_ROT as u8);
        CORNER_COLORS[corner_type as usize][rel_corner_rot as usize]
    }

    // Same idea as get_corner_face_color() but for edges
    fn get_edge_face_color(&self, edge_idx: usize, rotation_shift: u8) -> CubeColor {
        let edge_type = self.edge_types[edge_idx];
        let edge_rot = self.edge_rots[edge_idx];

        let rel_edge_rot = (edge_rot + rotation_shift) % (NUM_EDGE_ROT as u8);
        EDGE_COLORS[edge_type as usize][rel_edge_rot as usize]
    }

    pub fn get_facelet_color(&self, face: CubeColor, facelet_pos: FaceletPos) -> CubeColor {
        use FaceletPos::*;
        match facelet_pos {
            // Just return the face
            Center => face,

            TopLeft|TopRight|BottomLeft|BottomRight => {
                // Get corner
                let face_corner_idx = match facelet_pos {
                    TopLeft => 0,
                    TopRight => 1,
                    BottomRight => 2,
                    BottomLeft => 3,
                    _ => panic!()
                };

                let rotation_shift = match face {
                    CubeColor::White | CubeColor::Yellow => [0, 0, 0, 0],
                    _ => [2, 1, 2, 1],
                }[face_corner_idx];

                let corner_idx = FACE_CORNER_INDICES[face as usize][face_corner_idx];
                self.get_corner_face_color(corner_idx, rotation_shift)
            },
            TopMid|MidLeft|MidRight|BottomMid => {
                // Get edge
                let face_edge_idx = match facelet_pos {
                    TopMid => 0,
                    MidRight => 1,
                    BottomMid => 2,
                    MidLeft => 3,
                    _ => panic!()
                };

                // TODO: Why must we do this? :(
                let face_edge_idx_shift = match face {
                    CubeColor::White => 0,
                    CubeColor::Yellow => 3,
                    CubeColor::Red | CubeColor::Orange => 1,
                    CubeColor::Green => 1,
                    CubeColor::Blue => 1,
                };

                let face_edge_idx = (face_edge_idx + face_edge_idx_shift) % 4;

                let rotation_shift = match face {
                    CubeColor::White | CubeColor::Yellow => [0, 0, 0, 0],
                    _ => [1, 1, 1, 0]
                }[face_edge_idx];

                let edge_idx = FACE_EDGE_INDICES[face as usize][face_edge_idx];
                self.get_edge_face_color(edge_idx, rotation_shift)
            }
        }
    }

    pub fn get_all_bytes(&self) -> Vec<u8> {
        self.corner_types.into_iter()
            .chain(self.corner_rots.into_iter())
            .chain(self.edge_types.into_iter())
            .chain(self.edge_rots.into_iter())
            .collect::<Vec<u8>>()
    }

    // Group every 8 data bytes into a u64, then hash them all with almost-FNV
    pub fn calc_hash(&self) -> u64 {
        let all_bytes = self.get_all_bytes();
        debug_assert!( // Must be divisible by 8
            !all_bytes.is_empty() && (all_bytes.len() % 8 == 0)
        );

        // Ref: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        const FNV_START_VAL: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut result = FNV_START_VAL;
        for i in (0..all_bytes.len()).step_by(8) {
            let slice = &all_bytes[i..i + 8];
            let as_u64 = u64::from_le_bytes(slice.try_into().unwrap());

            result = result.wrapping_mul(FNV_PRIME);
            result ^= as_u64;
        }
        result
    }
}

impl Default for CubeState {
    fn default() -> Self { Self::solved() }
}

impl Display for CubeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        f.write_str("Cube display:\n")?;
        f.write_fmt(format_args!("\tCorner types: {:?}\n", self.corner_types))?;
        f.write_fmt(format_args!("\tEdge types: {:?}\n", self.edge_types))?;

        ///////////

        let color_to_str = |color: CubeColor| -> String {
            use CubeColor::*;
            const BASE_STR: &str = "\u{2592}\u{2592}"; // Extended-ASCII half-dither block character
            match color {
                White  => BASE_STR.bright_white().on_bright_white(),
                Red    => BASE_STR.bright_red().on_bright_red(),
                Blue   => BASE_STR.bright_blue().on_bright_blue(),
                Orange => BASE_STR.bright_yellow().on_bright_red(), // Combine red and yellow
                Green  => BASE_STR.bright_green().on_bright_green(),
                Yellow => BASE_STR.bright_yellow().on_bright_yellow(),
            }.to_string()
        };

        // Number format:
        // - First digit = CubeColor index
        // - Second digit = Facelet index
        const TEMPLATE_STR_ROWS: [&str; 13] = [
            "           +----------+",
            "           | 40 41 42 |",
            "           | 43 44 45 |",
            "           | 46 47 48 |",
            "+----------+----------+----------+----------+",
            "| 10 11 12 | 00 01 02 | 30 31 32 | 50 51 52 |",
            "| 13 14 15 | 03 04 05 | 33 34 35 | 53 54 55 |",
            "| 16 17 18 | 06 07 08 | 36 37 38 | 56 57 58 |",
            "+----------+----------+----------+----------+",
            "           | 20 21 22 |",
            "           | 23 24 25 |",
            "           | 26 27 28 |",
            "           +----------+",
        ];

        // Copy in template
        let mut output = String::new();
        for row in TEMPLATE_STR_ROWS {
            output += "\t";
            output += row;
            output += "\n";
        }

        // Replace template facelets with the ascii-colors
        for face_color in CubeColor::iter() {
            use CubeColor::*;

            // Make the set of facelets, rotating as needed (turns the display clockwise)
            let facelet_index_rot = match face_color {
                White  => 2,
                Red    => 2,
                Blue   => 1,
                Orange => 0,
                Green  => 3,
                Yellow => 2,
            };

            for facelet_pos in FaceletPos::iter() {

                // NOTE: We match to a leading space to prevent replacing color codes
                let replacement_key = format!(" {}{}", face_color as usize, facelet_pos as usize);

                // Rotate with this face's "facelet_index_rot"
                let mut mapped_facelet_pos = facelet_pos;
                for _ in 0..facelet_index_rot {
                    // Rotate mapped position backwards
                    mapped_facelet_pos = mapped_facelet_pos.rotate_counterclockwise();
                }

                let color = self.get_facelet_color(face_color, mapped_facelet_pos);
                let color_str = " ".to_string() + &color_to_str(color);
                output = output.replace(&replacement_key, &color_str);
            }
        }

        // Replace center colors
        output = output
            .replace("WW", &color_to_str(CubeColor::White))
            .replace("RR", &color_to_str(CubeColor::Red))
            .replace("OO", &color_to_str(CubeColor::Orange))
            .replace("GG", &color_to_str(CubeColor::Green))
            .replace("BB", &color_to_str(CubeColor::Blue))
            .replace("YY", &color_to_str(CubeColor::Yellow));

        f.write_str(&output)
    }
}