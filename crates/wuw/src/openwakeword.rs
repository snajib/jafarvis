// OpenWakeWord constants

// mel model produces 76 frames from 12,560 samples at 16kHz.
// both values empirically verified against melspectrogram.onnx v0.5.1
// and embedding_model.onnx v0.5.1 — the embedding model requires
// exactly 76 frames as input. do not change without re-verifying
// against both models.
pub const MEL_WINDOW_FRAMES: usize = 76;
pub const MEL_HOP_SAMPLES: usize = 160;
pub const MEL_WINDOW_SAMPLES: usize = 12_560;
//
pub const MEL_SLIDE_FRAMES: usize = 8;
pub const MEL_SLIDE_SAMPLES: usize = MEL_SLIDE_FRAMES * MEL_HOP_SAMPLES;

// normalization applied to mel output before the embedding model
pub const MEL_NORM_SCALE: f32 = 10.0;
pub const MEL_NORM_OFFSET: f32 = 2.0;

// mel filterbank bins, fixed by the embedding model architecture
pub const MEL_BINS: usize = 32;

// embedding model output dimension and keyword model sliding window depth.
// hey_jarvis keyword model input shape: [1, 16, 96]
pub const EMBEDDING_DIM: usize = 96;
pub const EMBEDDING_WINDOW: usize = 16;
