use apip::{
  capture::AudioSource,
  input::InputCapture,
  vad::{Vad, VadDecision},
};
use chrono::{DateTime, Local};
use std::time::SystemTime;

const CHUNK_DURATION_MS: u32 = 32;
const VAD_THRESHOLD: f32 = 0.2;
const TARGET_SAMPLE_RATE: u32 = 16000;
const DEFAULT_MODEL_PATH: &str = "models/silero_vad.onnx";

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("loading model...");
  let model_path =
    std::env::var("SILERO_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());

  let mut vad = Vad::new(&model_path, VAD_THRESHOLD)?;
  println!("model loaded, opening mic...");

  let mut source = InputCapture::new(CHUNK_DURATION_MS)?;
  let sample_rate = source.sample_rate;

  let chunk = source.read()?;
  let max = chunk
    .samples
    .iter()
    .cloned()
    .fold(f32::NEG_INFINITY, f32::max);
  println!("max amplitude: {:.4}", max);

  let mut resampler =
    apip::resample::Resampler::new(sample_rate, TARGET_SAMPLE_RATE, source.chunk_size)?;
  println!("mic open, starting loop...");
  println!("listening...");

  let mut last = VadDecision::Silence;

  loop {
    let chunk = source.read()?;
    let resampled = resampler.process(&chunk.samples)?;
    let decision = vad.process(&resampled, TARGET_SAMPLE_RATE)?;

    if decision != last {
      let now: DateTime<Local> = SystemTime::now().into();
      match decision {
        VadDecision::Speech => println!("[speech]  {}", now.format("%H:%M:%S")),
        VadDecision::Silence => println!("[silence] {}", now.format("%H:%M:%S")),
      }
      last = decision;
    }
  }
}
