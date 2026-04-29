use std::collections::VecDeque;

use ndarray::{Array1, Array3};
use ort::session::Session;
use ort::value::Value;

use apip::{AudioChunk, FRAME_SAMPLES, SAMPLE_RATE};

use crate::error::WuwError;
use crate::openwakeword::{
  EMBEDDING_DIM, EMBEDDING_WINDOW, MEL_BINS, MEL_NORM_OFFSET, MEL_NORM_SCALE, MEL_SLIDE_SAMPLES,
  MEL_WINDOW_FRAMES, MEL_WINDOW_SAMPLES,
};
use crate::trigger::{Trigger, TriggerSource};

pub struct WakeWord {
  mel_session: Session,
  embedding_session: Session,
  keyword_session: Session,

  // accumulates raw f32 samples until MEL_WINDOW_SAMPLES is reached.
  // then slides forward by MEL_SLIDE_SAMPLES after each inference
  audio_accumulator: Vec<f32>,

  // MEL_SLIDE_SAMPLES doesn't divide evenly into FRAME_SAMPLES.
  // accumulates sample debt and triggers slide when it crosses
  // MEL_SLIDE_SAMPLES
  slide_remainder: usize,

  // sliding window of last EMBEDDING_WINDOW embedding vectors
  embedding_buffer: VecDeque<[f32; EMBEDDING_DIM]>,

  // ring of last preroll_frames raw audio frames. carried in
  // trigger so ASR receives context audio that preceded activation
  preroll_buffer: VecDeque<Vec<f32>>,
  preroll_frames: usize,

  // estimated duration of the wake word in frames. used to tell
  // the chunker how much of the preroll to skip before passing
  // audio to ASR. tuned so the model fires slightly
  // after the wake word ends so this will need adjustment.
  wuw_frame_count: usize,

  // detection fires when keyword model output exceeds this value
  threshold: f32,
}

impl WakeWord {
  pub fn new(
    mel_model_path: &str,
    embedding_model_path: &str,
    keyword_model_path: &str,
    threshold: f32,
    preroll_ms: usize,
    wuw_duration_ms: usize,
  ) -> Result<Self, WuwError> {
    let preroll_frames = preroll_ms * SAMPLE_RATE as usize / (1_000 * FRAME_SAMPLES);
    let wuw_frame_count = wuw_duration_ms * SAMPLE_RATE as usize / (1_000 * FRAME_SAMPLES);

    Ok(Self {
      mel_session: load_session(mel_model_path)?,
      embedding_session: load_session(embedding_model_path)?,
      keyword_session: load_session(keyword_model_path)?,
      audio_accumulator: Vec::with_capacity(MEL_WINDOW_SAMPLES),
      embedding_buffer: VecDeque::with_capacity(EMBEDDING_WINDOW),
      slide_remainder: 0,
      threshold,
      preroll_frames,
      preroll_buffer: VecDeque::with_capacity(preroll_frames),
      wuw_frame_count,
    })
  }

  /// called by pipeline on every 512-sample frame. returns Some(Trigger)
  /// when the keyword model scores above threshold, None otherwise
  pub fn push_frame(&mut self, chunk: &apip::AudioChunk) -> Result<Option<Trigger>, WuwError> {
    // maintain pre-roll ring
    if self.preroll_buffer.len() == self.preroll_frames {
      self.preroll_buffer.pop_front();
    }
    self.preroll_buffer.push_back(chunk.samples.clone());

    self.audio_accumulator.extend_from_slice(&chunk.samples);
    self.slide_remainder += FRAME_SAMPLES;

    if self.audio_accumulator.len() < MEL_WINDOW_SAMPLES {
      return Ok(None);
    }

    let mel = self.run_mel()?;
    let embedding = self.run_embedding(&mel)?;

    if self.embedding_buffer.len() == EMBEDDING_WINDOW {
      self.embedding_buffer.pop_front();
    }
    self.embedding_buffer.push_back(embedding);

    if self.embedding_buffer.len() < EMBEDDING_WINDOW {
      self.slide_accumulator();
      return Ok(None);
    }

    let score = self.run_keyword()?;
    self.slide_accumulator();

    if score >= self.threshold {
      Ok(Some(self.build_trigger(TriggerSource::WakeWord)))
    } else {
      Ok(None)
    }
  }

  /// bypass inference and emit a trigger immediately from current
  /// pre-roll buffer. called by the PTT input handler
  pub fn push_to_talk(&self) -> Trigger {
    self.build_trigger(TriggerSource::PTT)
  }

  fn run_mel(&mut self) -> Result<ndarray::ArrayD<f32>, WuwError> {
    let window = &self.audio_accumulator[..MEL_WINDOW_SAMPLES];
    let input = Array1::from_vec(window.to_vec())
      .into_shape_with_order((1, MEL_WINDOW_SAMPLES))
      .map_err(|e| WuwError::FeatureExtraction(e.to_string()))?;

    let input_val =
      Value::from_array(input).map_err(|e| WuwError::FeatureExtraction(e.to_string()))?;

    let result = self
      .mel_session
      .run(ort::inputs!["input" => input_val])
      .map_err(|e| WuwError::FeatureExtraction(e.to_string()))?;

    let mel = result[0]
      .try_extract_array::<f32>()
      .map_err(|e| WuwError::FeatureExtraction(e.to_string()))?
      .into_owned();

    Ok(mel.mapv(|x| x / MEL_NORM_SCALE + MEL_NORM_OFFSET))
  }

  fn run_embedding(
    &mut self,
    mel: &ndarray::ArrayD<f32>,
  ) -> Result<[f32; EMBEDDING_DIM], WuwError> {
    let input = mel
      .view()
      .into_shape_with_order((MEL_WINDOW_FRAMES, MEL_BINS, 1))
      .map_err(|e| WuwError::EmbeddingInference(e.to_string()))?
      .to_owned(); // <- add this

    let input_val =
      Value::from_array(input).map_err(|e| WuwError::EmbeddingInference(e.to_string()))?;
    let result = self
      .embedding_session
      .run(ort::inputs!["input_1" => input_val])
      .map_err(|e| WuwError::EmbeddingInference(e.to_string()))?;

    let raw = result[0]
      .try_extract_array::<f32>()
      .map_err(|e| WuwError::EmbeddingInference(e.to_string()))?;

    let slice = raw
      .as_slice()
      .ok_or_else(|| WuwError::EmbeddingInference("embedding output not contiguous".to_string()))?;

    let mut embedding = [0f32; EMBEDDING_DIM];
    embedding.copy_from_slice(&slice[..EMBEDDING_DIM]);
    Ok(embedding)
  }

  fn run_keyword(&mut self) -> Result<f32, WuwError> {
    let flat: Vec<f32> = self.embedding_buffer.iter().flatten().copied().collect();
    let input = Array3::from_shape_vec((1, EMBEDDING_WINDOW, EMBEDDING_DIM), flat)
      .map_err(|e| WuwError::KeywordInference(e.to_string()))?;

    let input_val =
      Value::from_array(input).map_err(|e| WuwError::KeywordInference(e.to_string()))?;

    let result = self
      .keyword_session
      .run(ort::inputs!["input_1" => input_val])
      .map_err(|e| WuwError::KeywordInference(e.to_string()))?;

    let raw = result[0]
      .try_extract_array::<f32>()
      .map_err(|e| WuwError::KeywordInference(e.to_string()))?;

    raw
      .as_slice()
      .and_then(|s| s.first())
      .copied()
      .ok_or_else(|| WuwError::KeywordInference("empty keyword output".to_string()))
  }

  fn slide_accumulator(&mut self) {
    while self.slide_remainder >= MEL_SLIDE_SAMPLES {
      let drain = MEL_SLIDE_SAMPLES.min(self.audio_accumulator.len());
      self.audio_accumulator.drain(..drain);
      self.slide_remainder -= MEL_SLIDE_SAMPLES;
    }
  }

  fn build_trigger(&self, source: TriggerSource) -> Trigger {
    let preroll_samples: Vec<f32> = self.preroll_buffer.iter().flatten().copied().collect();

    Trigger {
      source,
      preroll: AudioChunk {
        samples: preroll_samples,
        sample_rate: SAMPLE_RATE,
        is_speech: true,
      },
      wuw_frame_count: self.wuw_frame_count,
    }
  }
}

fn load_session(path: &str) -> Result<Session, WuwError> {
  Session::builder()
    .map_err(|e| WuwError::ModelLoad {
      path: path.to_string(),
      reason: e.to_string(),
    })?
    .commit_from_file(path)
    .map_err(|e| WuwError::ModelLoad {
      path: path.to_string(),
      reason: e.to_string(),
    })
}
