use apip::{
  capture::{AudioChunk, AudioSource},
  chunker::Chunker,
  input::InputCapture,
  resample::Resampler,
  vad::{Vad, VadDecision},
};
use chrono::{DateTime, Local};
use std::time::SystemTime;
use tokio::sync::mpsc;

const CHUNK_DURATION_MS: u32 = 32;
const VAD_THRESHOLD: f32 = 0.35;
const TARGET_SAMPLE_RATE: u32 = 16000;
const DEFAULT_MODEL_PATH: &str = "models/silero_vad.onnx";
const TRAILING_SILENCE_MS: u32 = 1200;
const MAX_UTTERANCE_MS: u32 = 15_000;
const CHANNEL_CAPACITY: usize = 32;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("loading model...");
  let model_path =
    std::env::var("SILERO_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
  let mut vad = Vad::new(&model_path, VAD_THRESHOLD)?;

  println!("model loaded, opening mic...");
  let mut source = InputCapture::new(CHUNK_DURATION_MS)?;
  let mut resampler = Resampler::new(source.sample_rate, TARGET_SAMPLE_RATE, source.chunk_size)?;

  let (tx, rx) = mpsc::channel::<AudioChunk>(CHANNEL_CAPACITY);
  let mut chunker = Chunker::new(
    tx,
    TRAILING_SILENCE_MS,
    MAX_UTTERANCE_MS,
    TARGET_SAMPLE_RATE,
    CHUNK_DURATION_MS,
  );
  tokio::spawn(utterance_handler(rx));

  println!("listening...");
  let mut last = VadDecision::Silence;
  loop {
    let chunk = source.read()?;
    let resampled = resampler.process(&chunk.samples)?;
    let resampled_chunk = AudioChunk {
      samples: resampled,
      sample_rate: TARGET_SAMPLE_RATE,
    };
    let decision = vad.process(&resampled_chunk.samples, TARGET_SAMPLE_RATE)?;

    if decision != last {
      let now: DateTime<Local> = SystemTime::now().into();
      match decision {
        VadDecision::Speech => println!("[speech]  {}", now.format("%H:%M:%S")),
        VadDecision::Silence => println!("[silence] {}", now.format("%H:%M:%S")),
      }
      last = decision;
    }

    chunker.push(&resampled_chunk, decision)?;
  }
}

async fn utterance_handler(mut rx: mpsc::Receiver<AudioChunk>) {
  while let Some(utterance) = rx.recv().await {
    println!(
      "[utterance] {} samples ({:.1}s)",
      utterance.samples.len(),
      utterance.samples.len() as f32 / utterance.sample_rate as f32
    );
  }
}
