#[derive(Debug, thiserror::Error)]
pub enum ApipError {
  #[error("audio device error: {0}")]
  Device(String),
  #[error("stream error: {0}")]
  Stream(String),
}
