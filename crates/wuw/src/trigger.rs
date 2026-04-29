use apip::AudioChunk;

#[derive(Debug, Clone)]
pub enum TriggerSource {
  WakeWord,
  PTT,
}

#[derive(Debug, Clone)]
pub struct Trigger {
  pub source: TriggerSource,

  // full pre-roll buffer snapshot at the moment of activation.
  // contains wuw_frame_count frames of wake word audio followed
  // by any remaining frames up to the moment of detection
  pub preroll: AudioChunk,

  // number of frames at the start of pre-roll that are the
  // wake word itself. chunker skips before passing audio to ASR
  pub wuw_frame_count: usize,
}
