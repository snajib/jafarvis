use crate::error::ApipError;
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler as RubatoResampler};

pub struct Resampler {
  resampler: Fft<f32>,
  output_buf: Vec<f32>,
}

impl Resampler {
  pub fn new(
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_chunk_size: usize,
  ) -> Result<Self, ApipError> {
    let resampler = Fft::<f32>::new(
      input_sample_rate as usize,
      output_sample_rate as usize,
      input_chunk_size,
      1,
      1,
      FixedSync::Input,
    )
    .map_err(|e| ApipError::Device(e.to_string()))?;

    let output_len = resampler.output_frames_max();
    let output_buf = vec![0.0f32; output_len];

    Ok(Self {
      resampler,
      output_buf,
    })
  }

  pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, ApipError> {
    let n_frames = input.len();
    let input_adapter =
      InterleavedSlice::new(input, 1, n_frames).map_err(|e| ApipError::Stream(e.to_string()))?;

    let output_len = self.output_buf.len();
    let mut output_adapter = InterleavedSlice::new_mut(&mut self.output_buf, 1, output_len)
      .map_err(|e| ApipError::Stream(e.to_string()))?;

    let (_, output_frames) = self
      .resampler
      .process_into_buffer(&input_adapter, &mut output_adapter, None)
      .map_err(|e| ApipError::Stream(e.to_string()))?;

    Ok(self.output_buf[..output_frames].to_vec())
  }
}
