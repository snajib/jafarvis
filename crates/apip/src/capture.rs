use crate::error::ApipError;

pub struct AudioChunk {
  pub samples: Vec<f32>,
  pub sample_rate: u32,
}

pub trait AudioSource: Send {
  fn read(&mut self) -> Result<AudioChunk, ApipError>;
}
