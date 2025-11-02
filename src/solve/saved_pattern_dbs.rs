use crate::cube::{CubeMask, CubeState};
use crate::solve::PatternDB;

pub const DIR_BASE_PATH: &str = "./cubesolver_cache/pattern_dbs";

pub struct SavedPatternDB {
    pub name: String,
    pub db: PatternDB,
    pub mask: CubeMask,
    pub depth_limit: Option<u8>
}

impl SavedPatternDB {
    pub fn new(name: &str, mask: CubeMask, depth_limit: Option<u8>) -> Self {
        Self {
            name: name.to_string(),
            mask, depth_limit,
            db: PatternDB::default()
        }
    }

    // Returns (was hash found, depth to solve/max depth if not found)
    pub fn get_solution_depth(&self, cube_state: &CubeState) -> (bool, u8) {
        let masked_hash =  cube_state.calc_masked_sym_hash(&self.mask);
        if let Some(depth) = self.db.map.get(&masked_hash) {
            (true, *depth)
        } else {
            (false, self.db.max_depth + 1)
        }
    }

    pub fn get_save_path(&self) -> String {
        format!("{DIR_BASE_PATH}/{}.ptdb", self.name)
    }

    pub fn save(&self) {
        let _ = std::fs::create_dir_all(DIR_BASE_PATH); // Create dir if it doesn't exist
        self.db.save_to_file(self.get_save_path().as_str()).unwrap();
    }

    // Returns true if loaded
    pub fn load_or_make(&mut self) -> bool {
        let exists = std::fs::exists(self.get_save_path().as_str()).unwrap();
        if exists {
            self.db = PatternDB::load_from_file(self.get_save_path().as_str()).unwrap();
            if self.depth_limit.is_some() {
                assert_eq!(self.db.max_depth, self.depth_limit.unwrap());
            }
            true
        } else {
            println!("Pattern database \"{}\" not found!", self.name);
            self.db = PatternDB::build(
                |cube_state| -> u64 {
                    cube_state.calc_masked_sym_hash(&self.mask)
                },
                None, self.depth_limit
            );
            self.save();
            false
        }
    }
}

pub struct SavedPatternDBs {
    // n-depth database of the entire cube
    all: SavedPatternDB,

    // Databases of portions of the cube
    partials: Vec<SavedPatternDB>,
}

impl SavedPatternDBs {
    const CORNERS_MASK: CubeMask = CubeMask::from_byte_arrays(
        [1,1,1,1, 1,1,1,1],
        [0,0,0,0, 0,0,0,0, 0,0,0,0]
    );

    const EVEN_MASK: CubeMask = CubeMask::from_byte_arrays(
        [1,0,1,0, 1,0,1,0],
        [1,0,1,0, 1,0,1,0, 1,0,1,0]
    );

    const ODD_MASK: CubeMask = Self::EVEN_MASK.inv();

    pub fn load_make() -> SavedPatternDBs {
        println!("Making/loading saved pattern databases...");
        let mut result = SavedPatternDBs {
            all: SavedPatternDB::new("all", CubeMask::all(), Some(7)),
            partials: vec![
                SavedPatternDB::new("corners", Self::CORNERS_MASK, Some(9)),
                SavedPatternDB::new("evens", Self::EVEN_MASK, Some(7)),
                SavedPatternDB::new("odds", Self::ODD_MASK, Some(7)),
            ],
        };

        result.all.load_or_make();
        for db in result.partials.iter_mut() {
            db.load_or_make();
        }

        result
    }

    // Gets the lower bound to the number of moves needed to solve the cube
    pub fn get_solution_lower_bound(&self, cube_state: &CubeState) -> u8 {
        let (full_solution_found, full_min_moves) = self.all.get_solution_depth(cube_state);
        if full_solution_found {
            // Exact solution found
            full_min_moves
        } else {
            // We'll just take the minimum of the highest partial solution
            let mut min_moves = full_min_moves;
            for partial in &self.partials {
                let (_found, partial_min_moves) = partial.get_solution_depth(cube_state);
                if partial_min_moves > min_moves {
                    min_moves = partial_min_moves;
                }
            }

            min_moves
        }
    }
}