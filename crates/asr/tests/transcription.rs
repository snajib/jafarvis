use apip::AudioChunk;
use asr::{AsrError, SAMPLE_RATE, Transcriber};
use std::env;

fn model_path() -> String {
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  format!(
    "{}/../../vendor/whisper.cpp/models/ggml-base.bin",
    manifest_dir
  )
}

#[test]
fn loads_model() {
  let path = model_path();
  let result = Transcriber::new(&path);
  assert!(
    result.is_ok(),
    "failed to load model from {}: {:?}",
    path,
    result.err()
  );
}

#[test]
fn rejects_invalid_sample_rate() {
  let path = model_path();
  let mut transcriber = Transcriber::new(&path).unwrap();

  let audio = AudioChunk {
    samples: vec![0.0; 512],
    sample_rate: 48000, // wrong rate
    is_speech: true,
  };

  let result = transcriber.transcribe(&audio);
  assert!(matches!(result, Err(AsrError::InvalidSampleRate { .. })));
}

#[test]
fn transcribes_silence() {
  let path = model_path();
  let mut transcriber = Transcriber::new(&path).unwrap();

  // 1 second of silence at 16kHz
  let audio = AudioChunk {
    samples: vec![0.0; SAMPLE_RATE as usize],
    sample_rate: SAMPLE_RATE,
    is_speech: false,
  };

  let result = transcriber.transcribe(&audio);
  assert!(result.is_ok(), "transcription failed: {:?}", result.err());

  // silence typically produces empty or minimal output
  let text = result.unwrap();
  assert!(text.len() < 50, "unexpected output for silence: {:?}", text);
}

// NOTE: real audio fixtures would go here
// for now, testing with synthetic audio validates the FFI plumbing works
// future: add actual WAV files with known transcriptions for E2E validation
