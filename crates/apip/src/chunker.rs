use tokio::sync::mpsc::Sender;

use crate::capture::AudioChunk;
use crate::error::ApipError;
use crate::vad::VadDecision;

enum ChunkerState {
  Silence,
  Speaking,
  TrailingSilence,
}

pub struct Chunker {
  sender: Sender<AudioChunk>,
  buffer: Vec<f32>,
  state: ChunkerState,
  silence_frame_count: u32,
  utterance_frame_count: u32,
  trailing_silence_frames: u32,
  max_utterance_frames: u32,
  sample_rate: u32,
}

impl Chunker {
  pub fn new(
    sender: Sender<AudioChunk>,
    trailing_silence_ms: u32,
    max_utterance_ms: u32,
    sample_rate: u32,
    frame_duration_ms: u32,
  ) -> Self {
    Self {
      sender,
      buffer: Vec::new(),
      state: ChunkerState::Silence,
      silence_frame_count: 0,
      utterance_frame_count: 0,
      trailing_silence_frames: trailing_silence_ms / frame_duration_ms,
      max_utterance_frames: max_utterance_ms / frame_duration_ms,
      sample_rate,
    }
  }

  pub fn push(&mut self, chunk: &AudioChunk, decision: VadDecision) -> Result<(), ApipError> {
    match (&self.state, decision) {
      (ChunkerState::Silence, VadDecision::Speech) => {
        self.state = ChunkerState::Speaking;
        self.buffer.extend_from_slice(&chunk.samples);
        self.utterance_frame_count = 1;
      }
      (ChunkerState::Speaking, VadDecision::Speech) => {
        self.buffer.extend_from_slice(&chunk.samples);
        self.utterance_frame_count += 1;

        if self.utterance_frame_count >= self.max_utterance_frames {
          self.emit()?;
        }
      }
      (ChunkerState::Speaking, VadDecision::Silence) => {
        self.state = ChunkerState::TrailingSilence;
        self.buffer.extend_from_slice(&chunk.samples);
        self.silence_frame_count = 1;
        self.utterance_frame_count += 1;
      }
      (ChunkerState::TrailingSilence, VadDecision::Speech) => {
        self.state = ChunkerState::Speaking;
        self.buffer.extend_from_slice(&chunk.samples);
        self.silence_frame_count = 0;
        self.utterance_frame_count += 1;
      }
      (ChunkerState::TrailingSilence, VadDecision::Silence) => {
        self.silence_frame_count += 1;
        self.utterance_frame_count += 1;

        if self.silence_frame_count >= self.trailing_silence_frames {
          self.emit()?;
        }
      }
      (ChunkerState::Silence, VadDecision::Silence) => {}
    }

    Ok(())
  }

  fn emit(&mut self) -> Result<(), ApipError> {
    let samples = std::mem::take(&mut self.buffer);
    let chunk = AudioChunk {
      samples,
      sample_rate: self.sample_rate,
      is_speech: false,
    };

    self
      .sender
      .try_send(chunk)
      .map_err(|e| ApipError::Stream(e.to_string()))?;

    self.state = ChunkerState::Silence;
    self.silence_frame_count = 0;
    self.utterance_frame_count = 0;

    Ok(())
  }
}
