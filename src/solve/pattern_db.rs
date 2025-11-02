use rayon::iter::ParallelIterator;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::mem;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::prelude::IntoParallelRefIterator;
use crate::cube::{CubeMove, CubeState};

#[derive(Debug)]
pub struct PatternDB {
    pub map: HashMap<u64, u8>,
    pub max_depth: u8
}

impl Default for PatternDB {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            max_depth: 0
        }
    }
}

impl PatternDB {

    pub fn build(hash_fn: impl Fn(&CubeState) -> u64 + Sync, expected_size: Option<usize>, max_depth: Option<u8>) -> Self {
        println!("Building pattern database:");

        let start_time = Instant::now();

        let starting_state = CubeState::default();
        let starting_hash = hash_fn(&starting_state);

        let expected_size = expected_size.unwrap_or(10_000);
        let mut final_map = HashMap::with_capacity(expected_size);
        final_map.insert(starting_hash, 0);

        let mut states_map = HashMap::from([(starting_hash, starting_state)]);

        let mut cur_depth: u8 = 1;
        while !states_map.is_empty() && cur_depth <= max_depth.unwrap_or(u8::MAX) {
            let next_states_map_arc = Arc::new(Mutex::new(
                HashMap::<u64, CubeState>::new()
            ));

            states_map.par_iter() // Use par_iter() for parallel processing
                .for_each(|(_hash, state)| {
                    // Here we use a map to prevent duplicate states of the same "hash_fn()" hash
                    let mut turn_nexts = Vec::with_capacity(CubeMove::ALL_TURNS.len());
                    for mv in CubeMove::ALL_TURNS {
                        let next_state = state.do_move(mv);
                        let next_state_hash = hash_fn(&next_state);
                        turn_nexts.push((next_state_hash, next_state));
                    }

                    // Insert the new states into the shared map (if they aren't already in there
                    let mut next_states_map_guard = next_states_map_arc.lock().unwrap();
                    for (hash, state) in turn_nexts {
                        // If we already found this state in a previous search, we should also ignore it
                        if final_map.contains_key(&hash) { continue; }

                        next_states_map_guard.entry(hash).or_insert(state);
                    }
                });

            let mut next_states_map = next_states_map_arc.lock().unwrap();

            for next_state_hash in next_states_map.keys() {
                // This part must remain sequential to safely modify the shared 'map'
                let map_entry = final_map.entry(*next_state_hash);

                // Check if the entry is Occupied (already visited at a shallower depth)
                if matches!(map_entry, Entry::Occupied(_)) {
                    panic!("Entry was already occupied, this should not occur")
                }

                // Insert the new depth and collect the state for the next iteration
                map_entry.or_insert(cur_depth);
            }

            println!(" > Depth {cur_depth}, entries: {}, end states: {}...", final_map.len(), next_states_map.len());

            // We will progress "states" to the next states
            states_map = mem::take(&mut next_states_map);
            if states_map.is_empty() {
                break;
            }
            cur_depth += 1;
        }

        let elapsed_secs = start_time.elapsed().as_secs_f32();
        let entries_per_sec = (final_map.len() as f32 / elapsed_secs).ceil() as usize;
        println!(
            " > Done, found {} entries (rate: {}k/s)!",
            final_map.len(),
            entries_per_sec / 1000
        );

        let max_depth = *final_map.values().max().unwrap();
        PatternDB { map: final_map, max_depth }
    }

    pub fn is_valid(&self) -> bool {
        if self.map.is_empty() { return false }
        if *self.map.values().max().unwrap() != self.max_depth { return false; }
        true
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("Saving pattern database to {}...", path);
        assert!(self.is_valid());
        let mut file = File::create(path)?;

        // Write max depth
        file.write_all(&self.max_depth.to_le_bytes())?;

        // Write map
        file.write_all(&self.map.len().to_le_bytes())?;
        for (hash, num_moves) in self.map.iter() {
            file.write_all(&hash.to_le_bytes())?;
            file.write_all(&num_moves.to_le_bytes())?;
        }

        println!(" > Done, wrote {} bytes", file.stream_position().unwrap());
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading pattern database from {}...", path);
        let mut file = BufReader::new(File::open(path)?);

        let mut max_depth_bytes = [0; size_of::<u8>()];
        file.read_exact(&mut max_depth_bytes)?;
        let max_depth = u8::from_le_bytes(max_depth_bytes);

        let mut map_len_bytes = [0; size_of::<usize>()];
        file.read_exact(&mut map_len_bytes)?;
        let map_len = usize::from_le_bytes(map_len_bytes);

        // Read all entries
        const ENTRY_SIZE: usize = size_of::<u64>() + size_of::<u8>();
        let mut buffer = vec![0u8; map_len * ENTRY_SIZE];
        file.read_exact(&mut buffer)?;

        // Parse entries from buffer
        let mut map = HashMap::with_capacity(map_len);
        for chunk in buffer.chunks_exact(ENTRY_SIZE) {
            let hash = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let num_moves = chunk[8];
            map.insert(hash, num_moves);
        }

        println!(" > Done, number of entries: {map_len}, max depth: {max_depth}");
        Ok(PatternDB { map, max_depth })
    }
}