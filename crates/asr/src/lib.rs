mod error;
mod transcriber;
mod whisper_ffi;

pub use error::AsrError;
pub use transcriber::Transcriber;

// re-export SAMPLE_RATE for validation
pub use apip::SAMPLE_RATE;
