use crate::error::ApipError;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_SAMPLES: usize = 512;
pub const FRAME_DURATION_MS: usize = 1_000 * FRAME_SAMPLES / SAMPLE_RATE as usize; //32ms

#[derive(Debug, Clone)]
pub struct AudioChunk {
  pub samples: Vec<f32>,
  pub sample_rate: u32,
  pub is_speech: bool,
}

pub trait AudioSource: Send {
  fn read(&mut self) -> Result<AudioChunk, ApipError>;
}
