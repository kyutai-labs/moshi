// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#![allow(unused)]
use std::io::prelude::*;

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

pub fn write_pcm_as_wav<W: Write, S: Sample>(
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

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    use symphonia::core::audio::Signal;
    use symphonia::core::conv::FromSample;
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

pub(crate) fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};

    let src = std::fs::File::open(path)?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .expect("no supported audio tracks");
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .expect("unsupported codec");
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}

pub(crate) fn resample(pcm_in: &[f32], sr_in: usize, sr_out: usize) -> anyhow::Result<Vec<f32>> {
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

pub(crate) fn write_opus_header<W: std::io::Write>(w: &mut W) -> std::io::Result<()> {
    use byteorder::WriteBytesExt;

    // https://wiki.xiph.org/OggOpus#ID_Header
    w.write_all(b"OpusHead")?;
    w.write_u8(1)?; // version
    w.write_u8(1)?; // channel count
    w.write_u16::<byteorder::LittleEndian>(3840)?; // pre-skip
    w.write_u32::<byteorder::LittleEndian>(48000)?; //  sample-rate in Hz
    w.write_i16::<byteorder::LittleEndian>(0)?; // output gain Q7.8 in dB
    w.write_u8(0)?; // channel map
    Ok(())
}

pub(crate) fn write_opus_tags<W: std::io::Write>(w: &mut W) -> std::io::Result<()> {
    use byteorder::WriteBytesExt;

    // https://wiki.xiph.org/OggOpus#Comment_Header
    let vendor = "KyutaiMoshi";
    w.write_all(b"OpusTags")?;
    w.write_u32::<byteorder::LittleEndian>(vendor.len() as u32)?; // vendor string length
    w.write_all(vendor.as_bytes())?; // vendor string, UTF8 encoded
    w.write_u32::<byteorder::LittleEndian>(0u32)?; // number of tags
    Ok(())
}
