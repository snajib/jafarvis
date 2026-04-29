use apip::AudioChunk;
use hound::WavReader;
use wuw::wuw::WakeWord;

const FRAME_SAMPLES: usize = 512;

fn model_path(name: &str) -> String {
  let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
  format!("{}/../../models/{}", manifest, name)
}

fn wav_path(name: &str) -> String {
  let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
  format!("{}/tests/{}", manifest, name)
}

fn load_and_run(wav_name: &str) -> usize {
  let mut wuw = WakeWord::new(
    &model_path("melspectrogram.onnx"),
    &model_path("embedding_model.onnx"),
    &model_path("hey_jarvis_v0.1.onnx"),
    0.5,
    800,
    700,
  )
  .unwrap();

  let mut reader = WavReader::open(wav_path(wav_name)).unwrap();
  let spec = reader.spec();

  let samples: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Int => reader
      .samples::<i16>()
      .map(|s| s.unwrap() as f32 / i16::MAX as f32)
      .collect(),
    hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
  };

  let mut trigger_count = 0;

  for frame in samples.chunks(FRAME_SAMPLES) {
    let mut padded = frame.to_vec();
    if padded.len() < FRAME_SAMPLES {
      padded.resize(FRAME_SAMPLES, 0.0);
    }

    let chunk = AudioChunk {
      samples: padded,
      sample_rate: spec.sample_rate,
      is_speech: true,
    };

    if wuw.push_frame(&chunk).unwrap().is_some() {
      trigger_count += 1;
    }
  }

  trigger_count
}

#[test]
fn detects_wake_word() {
  let triggers = load_and_run("positive.wav");
  assert!(triggers > 0, "expected at least one trigger, got none");
}

#[test]
fn no_false_triggers() {
  let triggers = load_and_run("negative.wav");
  assert_eq!(triggers, 0, "expected no triggers, got {}", triggers);
}
