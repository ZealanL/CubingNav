use std::sync::{Arc, Mutex};
use ndarray::{Array, Array1, Axis, CowArray, IxDyn};
use ort::{InMemorySession, NdArrayExtensions};
use crate::cube::{CubeMask, CubeState};
use crate::ml;
use crate::ml::NUM_TOKENS_PER_CUBE_STATE;

fn apply_softmax(logits: &Array<f32, IxDyn>) -> Array<f32, IxDyn> {
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

    pub fn infer(&self, cube_state: &CubeState, mask: &CubeMask) -> f32 {
        let cube_tokens = ml::cube_to_tokens(cube_state, mask);
        let token_indices_reshaped_array = Array1::from(cube_tokens.to_vec()).into_shape(
            (1, NUM_TOKENS_PER_CUBE_STATE)
        ).unwrap().mapv(|x| x as i64);

        let ort_array_input = CowArray::from(token_indices_reshaped_array).into_dyn();
        let ort_inputs = vec![
            ort::Value::from_array(
                self.session.allocator(), &ort_array_input
            ).unwrap()
        ];

        let outputs = self.session.run(ort_inputs).unwrap();
        let output = outputs.first().unwrap();
        let output_logits = output.try_extract::<f32>().unwrap()
            .view().to_owned();

        let output_probs = apply_softmax(&output_logits);
        let probs_vec = output_probs.into_raw_vec();

        // calculate argumax
        let mut max_prob_idx = 0;
        for i in 1..probs_vec.len() {
            if probs_vec[i] > probs_vec[max_prob_idx] {
                max_prob_idx = i;
            }
        }

        let mut expectation = 0.0;
        for i in 0..probs_vec.len() {
            expectation += (i as f32) * probs_vec[i];
        }

        expectation
        //max_prob_idx as f32
    }
}