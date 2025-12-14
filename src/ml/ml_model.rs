use std::sync::{Arc, Mutex};
use ndarray::{Array, Array1, Axis, CowArray, Ix1, IxDyn};
use ort::{InMemorySession, NdArrayExtensions};
use crate::cube::{CubeMask, CubeMove, CubeState};
use crate::ml;
use crate::ml::NUM_TOKENS_PER_CUBE_STATE;

fn apply_softmax(logits: &Array<f32, Ix1>) -> Array<f32, Ix1> {
    // Make logits all <=0 to fix numeric stability issues
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let neg_logits = logits.mapv(|x| x - max_val);

    let exp_logits = neg_logits.mapv(|x| x.exp());
    let exp_sum = exp_logits.sum();

    exp_logits.mapv(|x| x / exp_sum)
}

pub struct MLModel {
    pub session: InMemorySession<'static>, // Changed from InMemorySession<'a>
}

pub struct MLModelOutput {
    pub move_probs: Vec<f32>,
    pub state_val: f32
}

impl MLModel {
    pub fn load_onnx(num_threads: usize) -> Result<MLModel, ort::OrtError> {
        use std::sync::Arc;
        use ort::environment::Environment;
        // 2. Import the correct types
        use ort::{GraphOptimizationLevel, LoggingLevel, session::SessionBuilder};

        // This is an &'static [u8]
        const ONNX_MODEL_BYTES: &[u8] = include_bytes!("../../checkpoint/model.onnx");

        println!("Creating ONNX environment...");
        let environment = Arc::new(
            Environment::builder()
                .with_name("models_env")
                .with_log_level(LoggingLevel::Verbose)
                .build()?
        );

        println!("Creating session with model...");
        // The SessionBuilder::new and its methods remain the same.
        // The library handles the lifetime internally as 'static since it's from memory.
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(num_threads as i16)?
            .with_inter_threads(num_threads as i16)?
            .with_model_from_memory(ONNX_MODEL_BYTES)?; // This implicitly gives the session a 'static lifetime

        println!(" > Inputs:  {:?}", session.inputs);
        println!(" > Outputs: {:?}", session.outputs);

        // 3. The return type and struct no longer need the lifetime specifier.
        Ok(MLModel {
            session
        })
    }

    // Returns the probability distribution over all moves

    pub fn infer(&self, cube_states: &Vec<CubeState>) -> Vec<MLModelOutput> {
        let num_parallel = cube_states.len();
        let mut all_cube_tokens = Vec::with_capacity(NUM_TOKENS_PER_CUBE_STATE * num_parallel);
        for cube_state in cube_states {
            all_cube_tokens.append(
                &mut ml::cube_to_tokens(cube_state, &CubeMask::all()).to_vec()
            )
        }
        let token_indices_reshaped_array = Array1::from(all_cube_tokens.to_vec()).into_shape(
            (num_parallel, NUM_TOKENS_PER_CUBE_STATE)
        ).unwrap().mapv(|x| x as i64);

        let ort_array_input = CowArray::from(token_indices_reshaped_array).into_dyn();
        let ort_inputs = vec![
            ort::Value::from_array(
                self.session.allocator(), &ort_array_input
            ).unwrap()
        ];

        let outputs = self.session.run(ort_inputs).unwrap();
        assert_eq!(outputs.len(), 2);
        let output_logits = outputs[0].try_extract::<f32>().unwrap()
            .view().to_owned();
        let output_values = outputs[1].try_extract::<f32>().unwrap()
            .view().to_owned();

        let mut all_policy_probs = Vec::new();
        for row in output_logits.rows() {
            let policy_probs = apply_softmax(&row.to_owned()).into_raw_vec();
            all_policy_probs.push(policy_probs);
        }
        let all_values = output_values.into_raw_vec();

        assert_eq!(all_policy_probs.len(), num_parallel);
        assert_eq!(all_values.len(), num_parallel);

        let mut result = Vec::new();
        for i in 0..num_parallel {
            result.push(
                MLModelOutput {
                    move_probs: all_policy_probs[i].clone(),
                    state_val: all_values[i]
                }
            )
        }

        result
    }
}