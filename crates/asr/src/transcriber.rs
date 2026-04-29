use crate::SAMPLE_RATE;
use crate::error::AsrError;
use crate::whisper_ffi;
use apip::AudioChunk;
use std::ffi::{CStr, CString};

/// automatic speech recognition transcriber using Whisper
///
/// wraps whisper.cpp FFI in safe, idiomatic interface that consumes AudioChunk
/// from the apip pipeline. whisper expects 16kHz mono f32 — exact match to pipeline output.
pub struct Transcriber {
  ctx: *mut whisper_ffi::whisper_context,
}

impl Transcriber {
  /// create a new transcriber by loading a whisper model
  ///
  /// # arguments
  /// * `model_path` - path to the GGML model file (e.g., "models/whisper-base.bin")
  ///
  /// # errors
  /// returns `AsrError::ModelLoad` if the model file cannot be loaded
  pub fn new(model_path: &str) -> Result<Self, AsrError> {
    let c_path = CString::new(model_path).map_err(|e| AsrError::ModelLoad {
      path: model_path.to_string(),
      reason: format!("invalid path: {}", e),
    })?;

    let ctx = unsafe { whisper_ffi::whisper_init_from_file(c_path.as_ptr()) };

    if ctx.is_null() {
      return Err(AsrError::ModelLoad {
        path: model_path.to_string(),
        reason: "whisper_init_from_file returned null".to_string(),
      });
    }

    Ok(Self { ctx })
  }

  /// transcribe an audio chunk to text
  ///
  /// # arguments
  /// * `audio` - AudioChunk from the pipeline (must be 16kHz mono f32)
  ///
  /// # errors
  /// * `AsrError::InvalidSampleRate` if audio is not 16kHz
  /// * `AsrError::TranscriptionFailed` if inference fails
  /// * `AsrError::SegmentExtraction` if text extraction fails
  pub fn transcribe(&mut self, audio: &AudioChunk) -> Result<String, AsrError> {
    // validate sample rate — whisper requires exactly 16kHz
    if audio.sample_rate != SAMPLE_RATE {
      return Err(AsrError::InvalidSampleRate {
        expected: SAMPLE_RATE,
        got: audio.sample_rate,
      });
    }

    // get default transcription parameters
    let mut params = unsafe {
      whisper_ffi::whisper_full_default_params(
        whisper_ffi::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY,
      )
    };

    // disable console output — we handle it ourselves
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;

    // run transcription on audio samples
    let result = unsafe {
      whisper_ffi::whisper_full(
        self.ctx,
        params,
        audio.samples.as_ptr(),
        audio.samples.len() as i32,
      )
    };

    if result != 0 {
      return Err(AsrError::TranscriptionFailed(format!(
        "whisper_full returned {}",
        result
      )));
    }

    // extract text from all segments
    let num_segments = unsafe { whisper_ffi::whisper_full_n_segments(self.ctx) };
    let mut text = String::new();

    for i in 0..num_segments {
      let c_str = unsafe { whisper_ffi::whisper_full_get_segment_text(self.ctx, i) };

      if c_str.is_null() {
        return Err(AsrError::SegmentExtraction(format!(
          "null pointer for segment {}",
          i
        )));
      }

      let segment = unsafe { CStr::from_ptr(c_str) }
        .to_str()
        .map_err(|e| AsrError::SegmentExtraction(format!("invalid UTF-8: {}", e)))?;

      text.push_str(segment);
    }

    Ok(text)
  }
}

// RAII cleanup — free whisper context on drop
impl Drop for Transcriber {
  fn drop(&mut self) {
    if !self.ctx.is_null() {
      unsafe {
        whisper_ffi::whisper_free(self.ctx);
      }
    }
  }
}

// transcriber is Send — whisper context is thread-safe for ownership transfer
// not Sync — whisper_full modifies internal state, no concurrent access
unsafe impl Send for Transcriber {}
