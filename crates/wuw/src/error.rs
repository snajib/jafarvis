use thiserror::Error;

#[derive(Debug, Error)]
pub enum WuwError {
  #[error("failed to load wake word model from {path}: {reason}")]
  ModelLoad { path: String, reason: String },

  #[error("feature extraction failed: {0}")]
  FeatureExtraction(String),

  #[error("embedding inference failed: {0}")]
  EmbeddingInference(String),

  #[error("keyword inference failed: {0}")]
  KeywordInference(String),

  #[error("model input shape mismatch: expected {expected}, got {got}")]
  ShapeMismatch { expected: String, got: String },
}
