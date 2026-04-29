use crate::capture::{AudioChunk, AudioSource};
use crate::error::ApipError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use std::sync::mpsc::{self, Receiver, SyncSender};

const CHANNEL_CAPACITY: usize = 64;

pub struct InputCapture {
  receiver: Receiver<Vec<f32>>,
  pub sample_rate: u32,
  pub chunk_size: usize,
  // not using but don't drop, prefixed with _
  _stream: Stream,
}

impl InputCapture {
  pub fn new(chunk_duration_ms: u32) -> Result<Self, ApipError> {
    let host = cpal::default_host();
    let device = host
      .default_input_device()
      .ok_or_else(|| ApipError::Device("no input device found".into()))?;

    let config = device
      .default_input_config()
      .map_err(|e| ApipError::Device(e.to_string()))?;

    let sample_rate = config.sample_rate();
    let channels = config.channels();
    let chunk_size = (sample_rate * chunk_duration_ms / 1000) as usize;

    let (tx, rx): (SyncSender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::sync_channel(CHANNEL_CAPACITY);

    let stream = Self::build_stream(&device, &config, tx, channels)?;
    stream
      .play()
      .map_err(|e| ApipError::Stream(e.to_string()))?;

    Ok(Self {
      receiver: rx,
      sample_rate,
      chunk_size,
      _stream: stream,
    })
  }

  fn build_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    tx: SyncSender<Vec<f32>>,
    channels: u16,
  ) -> Result<Stream, ApipError> {
    let stream_config = config.config();

    let stream = match config.sample_format() {
      SampleFormat::F32 => device.build_input_stream(
        &stream_config,
        move |data: &[f32], _| {
          let mono = downmix(data, channels);
          let _ = tx.try_send(mono);
        },
        |e| eprintln!("stream error: {e}"),
        None,
      ),
      SampleFormat::I16 => device.build_input_stream(
        &stream_config,
        move |data: &[i16], _| {
          let float: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
          let mono = downmix(&float, channels);
          let _ = tx.try_send(mono);
        },
        |e| eprintln!("stream error: {e}"),
        None,
      ),
      fmt => {
        return Err(ApipError::Device(format!(
          "unsupported sample format: {fmt:?}"
        )));
      }
    }
    .map_err(|e| ApipError::Stream(e.to_string()))?;

    Ok(stream)
  }
}

impl AudioSource for InputCapture {
  fn read(&mut self) -> Result<AudioChunk, ApipError> {
    let mut samples = Vec::with_capacity(self.chunk_size);

    while samples.len() < self.chunk_size {
      let batch = self
        .receiver
        .recv()
        .map_err(|e| ApipError::Stream(e.to_string()))?;
      samples.extend(batch);
    }

    Ok(AudioChunk {
      samples,
      sample_rate: self.sample_rate,
      is_speech: false,
    })
  }
}

fn downmix(samples: &[f32], channels: u16) -> Vec<f32> {
  if channels == 1 {
    return samples.to_vec();
  }
  samples
    .chunks_exact(channels as usize)
    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
    .collect()
}
