// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io::prelude::*;
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: usize = 24_000;

pub(crate) struct AudioOutputData_ {
    resampled_data: std::collections::VecDeque<f32>,
    resampler: rubato::FastFixedIn<f32>,
    output_buffer: Vec<f32>,
    input_buffer: Vec<f32>,
    input_len: usize,
    // The number of (resampled) samples that have been seen so far.
    total_samples: usize,
    // Some subtitle index together with the index at which it should get printed.
    subs: VecDeque<(usize, String)>,
    mean_squares: f32,
}

impl AudioOutputData_ {
    pub(crate) fn new(input_sample_rate: usize, output_sample_rate: usize) -> Result<Self> {
        use rubato::Resampler;

        let resampled_data = std::collections::VecDeque::with_capacity(output_sample_rate * 10);
        let resample_ratio = output_sample_rate as f64 / input_sample_rate as f64;
        let resampler = rubato::FastFixedIn::new(
            resample_ratio,
            f64::max(resample_ratio, 1.0),
            rubato::PolynomialDegree::Septic,
            1024,
            1,
        )?;
        let input_buffer = resampler.input_buffer_allocate(true).remove(0);
        let output_buffer = resampler.output_buffer_allocate(true).remove(0);
        Ok(Self {
            resampled_data,
            resampler,
            input_buffer,
            output_buffer,
            input_len: 0,
            total_samples: 0,
            subs: VecDeque::new(),
            mean_squares: 0.,
        })
    }

    pub(crate) fn total_samples(&self) -> usize {
        self.total_samples
    }

    pub(crate) fn samples_in_buffer(&self) -> usize {
        self.resampled_data.len()
    }

    pub(crate) fn take_all(&mut self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.resampled_data.len());
        while let Some(elem) = self.resampled_data.pop_back() {
            data.push(elem);
        }
        data
    }

    pub(crate) fn db10(&self) -> f32 {
        10. + (self.mean_squares + 1e-10).log10()
    }

    // Assumes that the input buffer is large enough.
    fn push_input_buffer(&mut self, samples: &[f32]) {
        self.input_buffer[self.input_len..self.input_len + samples.len()].copy_from_slice(samples);
        self.input_len += samples.len();
        self.total_samples += samples.len();
    }

    pub(crate) fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        use rubato::Resampler;

        let mut pos_in = 0;
        loop {
            let rem = self.input_buffer.len() - self.input_len;
            let pos_end = usize::min(pos_in + rem, samples.len());
            self.push_input_buffer(&samples[pos_in..pos_end]);
            pos_in = pos_end;
            if self.input_len < self.input_buffer.len() {
                break;
            }
            let (_, out_len) = self.resampler.process_into_buffer(
                &[&self.input_buffer],
                &mut [&mut self.output_buffer],
                None,
            )?;
            for &elem in self.output_buffer[..out_len].iter() {
                self.resampled_data.push_front(elem)
            }
            self.input_len = 0;
        }
        Ok(())
    }
}

type AudioOutputData = Arc<Mutex<AudioOutputData_>>;

pub(crate) fn setup_output_stream(real_time: bool) -> Result<(cpal::Stream, AudioOutputData)> {
    setup_output_stream_map(real_time, |s| {
        use std::io::Write;
        print!("{s}");
        let _ = std::io::stdout().flush();
    })
}

pub(crate) fn setup_output_stream_map<F: FnMut(String) + Send + 'static>(
    real_time: bool,
    mut f: F,
) -> Result<(cpal::Stream, AudioOutputData)> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    println!("Setup audio output stream!");
    let host = cpal::default_host();
    let device = host.default_output_device().context("no output device available")?;
    let mut supported_configs_range = device.supported_output_configs()?;
    let config_range = match supported_configs_range.find(|c| c.channels() == 1) {
        // On macOS, it's commonly the case that there are only stereo outputs.
        None => device.supported_output_configs()?.next().context("no audio output available")?,
        Some(config_range) => config_range,
    };
    let sample_rate = cpal::SampleRate(SAMPLE_RATE as u32)
        .clamp(config_range.min_sample_rate(), config_range.max_sample_rate());
    let config: cpal::StreamConfig = config_range.with_sample_rate(sample_rate).into();
    let channels = config.channels as usize;
    println!(
        "cpal device: {} {} {config:?}",
        device.name().unwrap_or_else(|_| "unk".to_string()),
        config.sample_rate.0
    );
    let audio_data =
        Arc::new(Mutex::new(AudioOutputData_::new(SAMPLE_RATE, config.sample_rate.0 as usize)?));
    let ad = audio_data.clone();
    let mut total_samples = 0;
    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            data.fill(0.);
            let mut ad = ad.lock().unwrap();
            let mut last_elem = 0f32;
            loop {
                let should_pop = match ad.subs.front() {
                    None => false,
                    Some((i, _)) => *i < total_samples,
                };
                if !should_pop {
                    break;
                }
                if let Some((_, s)) = ad.subs.pop_front() {
                    f(s)
                }
            }
            for (idx, elem) in data.iter_mut().enumerate() {
                if idx % channels == 0 {
                    match ad.resampled_data.pop_back() {
                        None => break,
                        Some(v) => {
                            last_elem = v;
                            total_samples += 1;
                            *elem = v
                        }
                    }
                } else {
                    *elem = last_elem
                }
            }
            if real_time && ad.resampled_data.len() > SAMPLE_RATE / 2 {
                let pcm_data = ad.resampled_data.drain(..).collect::<Vec<_>>();
                if let Ok(pcm_data) = resample(&pcm_data, SAMPLE_RATE, SAMPLE_RATE * 2 / 3) {
                    for v in pcm_data.into_iter().rev() {
                        ad.resampled_data.push_back(v)
                    }
                }
            }
        },
        move |err| eprintln!("cpal error: {err}"),
        None, // None=blocking, Some(Duration)=timeout
    )?;
    stream.play()?;
    Ok((stream, audio_data))
}

pub(crate) fn setup_input_stream() -> Result<(cpal::Stream, AudioOutputData)> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    println!("Setup audio input stream!");
    let host = cpal::default_host();
    let device = host.default_input_device().context("no input device available")?;
    let mut supported_configs_range = device.supported_input_configs()?;
    let config_range = match supported_configs_range.find(|c| c.channels() == 1) {
        Some(config) => config,
        None => {
            let supported_configs = device.supported_input_configs()?.collect::<Vec<_>>();
            anyhow::bail!("no audio input available, supported configs {supported_configs:?}",)
        }
    };
    let sample_rate = cpal::SampleRate(SAMPLE_RATE as u32)
        .clamp(config_range.min_sample_rate(), config_range.max_sample_rate());
    let config: cpal::StreamConfig = config_range.with_sample_rate(sample_rate).into();
    println!(
        "cpal device: {} {} {config:?}",
        device.name().unwrap_or_else(|_| "unk".to_string()),
        config.sample_rate.0
    );
    let audio_data =
        Arc::new(Mutex::new(AudioOutputData_::new(config.sample_rate.0 as usize, SAMPLE_RATE)?));
    let ad = audio_data.clone();
    let sample_rate = config.sample_rate.0 as f32;
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut ad = ad.lock().unwrap();
            if !data.is_empty() {
                let l = data.len() as f32;
                let mean = data.iter().sum::<f32>() / l;
                let mean_squares = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l;
                let decay = (-l / sample_rate * 10.).exp2();
                ad.mean_squares = decay * ad.mean_squares + (1. - decay) * mean_squares;
            }
            if let Err(err) = ad.push_samples(data) {
                eprintln!("error processing audio input {err:?}")
            }
        },
        move |err| eprintln!("cpal error: {err}"),
        None, // None=blocking, Some(Duration)=timeout
    )?;
    stream.play()?;
    Ok((stream, audio_data))
}

pub(crate) fn resample(pcm_in: &[f32], sr_in: usize, sr_out: usize) -> Result<Vec<f32>> {
    use rubato::Resampler;

    let mut pcm_out =
        Vec::with_capacity((pcm_in.len() as f64 * sr_out as f64 / sr_in as f64) as usize + 1024);

    let mut resampler = rubato::FftFixedInOut::<f32>::new(sr_in, sr_out, 1024, 1)?;
    let mut output_buffer = resampler.output_buffer_allocate(true);
    let mut pos_in = 0;
    while pos_in + resampler.input_frames_next() < pcm_in.len() {
        let (in_len, out_len) =
            resampler.process_into_buffer(&[&pcm_in[pos_in..]], &mut output_buffer, None)?;
        pos_in += in_len;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    if pos_in < pcm_in.len() {
        let (_in_len, out_len) = resampler.process_partial_into_buffer(
            Some(&[&pcm_in[pos_in..]]),
            &mut output_buffer,
            None,
        )?;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    Ok(pcm_out)
}

pub trait Sample {
    fn to_i16(&self) -> i16;
}

impl Sample for f32 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for f64 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for i16 {
    fn to_i16(&self) -> i16 {
        *self
    }
}

pub(crate) fn write_pcm_as_wav<W: Write, S: Sample>(
    w: &mut W,
    samples: &[S],
    sample_rate: u32,
) -> std::io::Result<()> {
    let len = 12u32; // header
    let len = len + 24u32; // fmt
    let len = len + samples.len() as u32 * 2 + 8; // data
    let n_channels = 1u16;
    let bytes_per_second = sample_rate * 2 * n_channels as u32;
    w.write_all(b"RIFF")?;
    w.write_all(&(len - 8).to_le_bytes())?; // total length minus 8 bytes
    w.write_all(b"WAVE")?;

    // Format block
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // block len minus 8 bytes
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&n_channels.to_le_bytes())?; // one channel
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&bytes_per_second.to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?; // 2 bytes of data per sample
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // Data block
    w.write_all(b"data")?;
    w.write_all(&(samples.len() as u32 * 2).to_le_bytes())?;
    for sample in samples.iter() {
        w.write_all(&sample.to_i16().to_le_bytes())?
    }
    Ok(())
}
