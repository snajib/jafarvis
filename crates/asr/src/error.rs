#[derive(Debug, thiserror::Error)]
pub enum AsrError {
  #[error("failed to load model from {path}: {reason}")]
  ModelLoad { path: String, reason: String },

  #[error("failed to create whisper state: {0}")]
  StateCreation(String),

  #[error("transcription failed: {0}")]
  TranscriptionFailed(String),

  #[error("invalid sample rate: expected {expected} Hz, got {got} Hz")]
  InvalidSampleRate { expected: u32, got: u32 },

  #[error("failed to extract segment text: {0}")]
  SegmentExtraction(String),
}
