use std::fmt::{Display, Formatter};
use colored::Colorize;
use crate::cube::{CubeFace, CubeState, NUM_CORNERS, NUM_CORNER_ROT, NUM_EDGES, NUM_EDGE_ROT, NUM_FACELETS, NUM_FACELETS_PER_FACE};

use CubeFace::*;

const CORNER_COLORS: [[CubeFace; NUM_CORNER_ROT]; NUM_CORNERS] = {
    [
        [U, R, F],
        [U, F, L],
        [U, L, B],
        [U, B, R],

        [D, F, R],
        [D, L, F],
        [D, B, L],
        [D, R, B],
    ]
};

const EDGE_COLORS: [[CubeFace; NUM_EDGE_ROT]; NUM_EDGES] = {
    [
        [U, R],
        [U, F],
        [U, L],
        [U, B],

        [D, R],
        [D, F],
        [D, L],
        [D, B],

        [F, R],
        [F, L],
        [B, L],
        [B, R],
    ]
};

fn face_to_color_str(face: CubeFace) -> String {
    use CubeFace::*;

    const BASE_STR: &str = "\u{2592}\u{2592}"; // Extended-ASCII half-dither block character
    match face {
        U => BASE_STR.bright_white().on_bright_white(),
        D => BASE_STR.bright_yellow().on_bright_yellow(),

        R => BASE_STR.bright_yellow().on_bright_red(), // Combine red and yellow
        L => BASE_STR.bright_red().on_bright_red(),

        F => BASE_STR.bright_blue().on_bright_blue(),
        B => BASE_STR.bright_green().on_bright_green(),

    }.to_string()
}

impl Display for CubeState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("\n")?;
        f.write_fmt(format_args!("Corner permutations: {:?}\n", self.corn_perm))?;
        f.write_fmt(format_args!("Edge permutations: {:?}\n", self.edge_perm))?;
        f.write_fmt(format_args!("Corner rotations: {:?}\n", self.corn_rot))?;
        f.write_fmt(format_args!("Edge rotations: {:?}\n", self.edge_rot))?;

        // Format:
        // - First char: face
        // - Second digit: facelet index (0-9)
        const TEMPLATE_STR_ROWS: [&str; 13] = [
            "           +----------+",
            "           | B8 B7 B6 |",
            "           | B5 B4 B3 |",
            "           | B2 B1 B0 |",
            "+----------+----------+----------+----------+",
            "| L6 L3 L0 | U0 U1 U2 | R2 R5 R8 | D8 D7 D6 |",
            "| L7 L4 L1 | U3 U4 U5 | R1 R4 R7 | D5 D4 D3 |",
            "| L8 L5 L2 | U6 U7 U8 | R0 R3 R6 | D2 D1 D0 |",
            "+----------+----------+----------+----------+",
            "           | F0 F1 F2 |",
            "           | F3 F4 F5 |",
            "           | F6 F7 F8 |",
            "           +----------+",
        ];

        // Copy in template
        let mut output = String::new();
        for row in TEMPLATE_STR_ROWS {
            output += "\t";
            output += row;
            output += "\n";
        }

        // Map corner permutations to corner positions (same with rotations)
        let mut corners = [(0, 0); NUM_CORNERS];
        for i in 0..NUM_CORNERS {
            corners[self.corn_perm[i] as usize] = (i, self.corn_rot[i] as usize);
        }

        // Same with edges
        let mut edges = [(0, 0); NUM_EDGES];
        for i in 0..NUM_EDGES {
            edges[self.edge_perm[i] as usize] = (i, self.edge_rot[i] as usize);
        }

        let mut facelet_colors = [U; NUM_FACELETS];
        { // Map-in corner and edge facelet values

            // Ref: https://github.com/efrantar/rob-twophase/blob/master/src/face.h
            const CORNER_FACELET_MAP: [[usize; NUM_CORNER_ROT]; NUM_CORNERS] = [
                [ 8,  9, 20], [ 6, 18, 38], [ 0, 36, 47], [ 2, 45, 11],
                [29, 26, 15], [27, 44, 24], [33, 53, 42], [35, 17, 51]
            ];
            const EDGE_FACELET_MAP: [[usize; NUM_EDGE_ROT]; NUM_EDGES] = [
                [ 5, 10], [ 7, 19], [ 3, 37], [ 1, 46],
                [32, 16], [28, 25], [30, 43], [34, 52],
                [23, 12], [21, 41], [50, 39], [48, 14]
            ];

            for i in 0..NUM_CORNERS {
                for j in 0..NUM_CORNER_ROT {
                    let facelet_val_idx = CORNER_FACELET_MAP[i][j];

                    let (corner_type, corner_rot) = corners[i];
                    let facelet_color = CORNER_COLORS[corner_type][(corner_rot + j) % NUM_CORNER_ROT];
                    facelet_colors[facelet_val_idx] = facelet_color;
                }
            }
            
            for i in 0..NUM_EDGES {
                for j in 0..NUM_EDGE_ROT {
                    let facelet_val_idx = EDGE_FACELET_MAP[i][j];

                    let (edge_type, edge_rot) = edges[i];
                    let facelet_color = EDGE_COLORS[edge_type][(edge_rot + j) % NUM_EDGE_ROT];
                    facelet_colors[facelet_val_idx] = facelet_color;
                }
            }

            // Set centers
            for i in 0..CubeFace::COUNT {
                facelet_colors[i * NUM_FACELETS_PER_FACE + 4] = CubeFace::from(i);
            }
        }

        // Apply the mapped facelet colors to the template string
        for face in CubeFace::ALL {
            let face_idx = face as usize;
            for facelet_face_idx in 0..NUM_FACELETS_PER_FACE {
                // Index in the facelet colors array
                let facelet_idx = facelet_face_idx + (face_idx * NUM_FACELETS_PER_FACE);

                output = output.replace(
                    &format!("{face}{facelet_face_idx}"),
                    &face_to_color_str(facelet_colors[facelet_idx])
                );
            }
        }

        f.write_str(output.as_str())
    }
}