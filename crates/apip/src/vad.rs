use crate::error::ApipError;
use ort::session::Session;

pub struct Vad {
  session: Session,
  state: Vec<f32>, // shape [2,1,64]
  threshold: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
  Speech,
  Silence,
}

impl Vad {
  pub fn new(model_path: impl AsRef<std::path::Path>, threshold: f32) -> Result<Self, ApipError> {
    ort::init().with_name("jafarvis").commit();

    let session = Session::builder()
      .map_err(|e| ApipError::Device(e.to_string()))?
      .commit_from_file(model_path)
      .map_err(|e| ApipError::Device(e.to_string()))?;

    Ok(Self {
      session,
      state: vec![0.0f32; 2 * 1 * 128],
      threshold,
    })
  }

  pub fn reset(&mut self) {
    self.state.fill(0.0);
  }

  pub fn process(&mut self, chunk: &[f32], sample_rate: u32) -> Result<VadDecision, ApipError> {
    use ndarray::{Array0, Array2, Array3};
    use ort::value::TensorRef;

    let sr = Array0::from_elem([], sample_rate as i64);

    let input = Array2::from_shape_vec([1, chunk.len()], chunk.to_vec())
      .map_err(|e| ApipError::Stream(e.to_string()))?;

    let state = Array3::from_shape_vec([2, 1, 128], self.state.clone())
      .map_err(|e| ApipError::Stream(e.to_string()))?;

    let outputs = self
      .session
      .run(ort::inputs![
          "input" => TensorRef::from_array_view(input.view())
              .map_err(|e| ApipError::Stream(e.to_string()))?,
          "state" => TensorRef::from_array_view(state.view())
              .map_err(|e| ApipError::Stream(e.to_string()))?,
          "sr" => TensorRef::from_array_view(sr.view())
              .map_err(|e| ApipError::Stream(e.to_string()))?
      ])
      .map_err(|e| ApipError::Stream(e.to_string()))?;

    self.state = outputs["stateN"]
      .try_extract_array::<f32>()
      .map_err(|e| ApipError::Stream(e.to_string()))?
      .as_slice()
      .unwrap()
      .to_vec();

    let prob = outputs["output"]
      .try_extract_array::<f32>()
      .map_err(|e| ApipError::Stream(e.to_string()))?
      .as_slice()
      .unwrap()[0];

    Ok(if prob >= self.threshold {
      VadDecision::Speech
    } else {
      VadDecision::Silence
    })
  }
}
