// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// The legacy T5 based TTS.
#![allow(unused)]
use anyhow::{Error as E, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct SpeakerConfig {
    pub(crate) name: String,
    audio_sample: String,
    offset_s: f64,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Args {
    pub t5_config_file: Option<String>,
    pub t5_model_file: Option<String>,
    pub t5_tokenizer_file: Option<String>,
    pub lm_model_file: String,
    pub speakers: Vec<SpeakerConfig>,
    pub mimi_model_file: String,
    pub speaker_model_file: Option<String>,
}

impl Args {
    fn speaker_cond(&self) -> bool {
        !self.speakers.is_empty()
    }
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

pub(crate) fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
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
struct RenameVb(VarBuilder<'static>);

impl candle_nn::var_builder::SimpleBackend for RenameVb {
    fn get(
        &self,
        s: candle::Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: candle::DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        if name == "text_emb.weight" || name == "text_linear.weight" {
            return Tensor::zeros(s, dtype, dev);
        }

        let slices: Vec<&str> = name.split('.').collect();
        let name = if slices.len() >= 3 && slices[0] == "depformer" {
            let idx: usize = slices[1].parse()?;
            if slices[2] == "transformer" {
                let name = format!("depformer.{}", slices[3..].join("."));
                std::borrow::Cow::Owned(name)
            } else if slices[2] == "emb" {
                if idx == 0 {
                    return Tensor::zeros(s, dtype, dev);
                }
                let idx = idx - 1;
                let name = format!("depformer_emb.{idx}.{}", slices[3..].join("."));
                std::borrow::Cow::Owned(name)
            } else if slices[2] == "linear_in" {
                let name = format!("depformer_in.{idx}.{}", slices[3..].join("."));
                std::borrow::Cow::Owned(name)
            } else if slices[2] == "linear_out" {
                let name = format!("linears.{idx}.{}", slices[3..].join("."));
                std::borrow::Cow::Owned(name)
            } else {
                std::borrow::Cow::Borrowed(name)
            }
        } else {
            std::borrow::Cow::Borrowed(name)
        };
        self.0.get_with_hints_dtype(s, name.as_ref(), h, dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn speaker_pcm(
    mimi_sample_rate: f64,
    speaker_cond_duration_s: f64,
    speaker: &str,
    delay: f64,
    dev: &Device,
) -> Result<Tensor> {
    let (pcm, sample_rate) = pcm_decode(speaker)?;
    let pcm = if sample_rate != mimi_sample_rate as u32 {
        resample(&pcm, sample_rate as usize, mimi_sample_rate as usize)?
    } else {
        pcm
    };
    let start_pos = (delay * mimi_sample_rate) as usize;
    let sample_len = (speaker_cond_duration_s * mimi_sample_rate) as usize;
    let pcm = &pcm[start_pos..start_pos + sample_len];
    let pcm = Tensor::new(pcm, dev)?.reshape((1, 1, ()))?;
    Ok(pcm)
}

#[derive(Clone)]
pub(crate) struct Speaker {
    #[allow(unused)]
    name: String,
    pcm: Tensor,
}

#[derive(Clone)]
pub(crate) struct Model {
    speakers: Vec<Speaker>,
    model: moshi::tts::Model,
    mimi: moshi::mimi::Mimi,
    tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    dev: Device,
}

impl Model {
    pub(crate) fn new(args: &Args, dev: &Device) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = {
            if args.speaker_cond() {
                let repo = hf_hub::Repo::with_revision(
                    "google/mt5-large".into(),
                    hf_hub::RepoType::Model,
                    "refs/pr/2".into(),
                );
                api.repo(repo)
            } else {
                api.model("t5-large".into())
            }
        };
        let t5_config_filename = match &args.t5_config_file {
            None => repo.get("config.json")?,
            Some(f) => f.into(),
        };
        let t5_tokenizer_filename = match &args.t5_tokenizer_file {
            None => {
                if args.speaker_cond() {
                    api.model("lmz/mt5-tokenizers".into()).get("mt5-large.tokenizer.json")?
                } else {
                    repo.get("tokenizer.json")?
                }
            }
            Some(f) => f.into(),
        };
        let t5_weights_filename = match &args.t5_model_file {
            None => repo.get("model.safetensors")?,
            Some(f) => f.into(),
        };
        let t5_config = std::fs::read_to_string(t5_config_filename)?;
        let t5_config: candle_transformers::models::t5::Config = serde_json::from_str(&t5_config)?;
        let config = moshi::tts::Config::v0_2(t5_config);
        let tokenizer = tokenizers::Tokenizer::from_file(t5_tokenizer_filename).map_err(E::msg)?;
        let tokenizer = std::sync::Arc::new(tokenizer);

        // Load the model and process the data so that the model gets released immediately.
        let vb_t5 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[t5_weights_filename], candle::DType::F32, dev)?
        };
        let vb_lm = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&args.lm_model_file], candle::DType::BF16, dev)?
        };
        let (speakers, vb_speaker_cond) = if args.speakers.is_empty() {
            (vec![], None)
        } else {
            let model_file = args
                .speaker_model_file
                .as_ref()
                .map_or(args.mimi_model_file.as_str(), |v| v.as_str());
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_file], candle::DType::F32, dev)?
            };
            let mut speakers = Vec::with_capacity(args.speakers.len());
            for speaker in args.speakers.iter() {
                let pcm = speaker_pcm(
                    config.mimi.sample_rate,
                    config.speaker_cond_duration_s,
                    &speaker.audio_sample,
                    speaker.offset_s,
                    dev,
                )?;
                speakers.push(Speaker { pcm, name: speaker.name.to_string() })
            }
            (speakers, Some(vb))
        };
        let vb_lm =
            VarBuilder::from_backend(Box::new(RenameVb(vb_lm)), candle::DType::BF16, dev.clone());
        let model = moshi::tts::Model::new(&config, vb_t5, vb_lm, vb_speaker_cond)?;
        let mimi = moshi::mimi::load(&args.mimi_model_file, None, dev)?;
        Ok(Self { speakers, mimi, model, tokenizer, dev: dev.clone() })
    }

    pub(crate) fn run(
        &mut self,
        prompt: &str,
        speaker_index: usize,
        cfg_alpha: f64,
        lp: candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<u8>> {
        let dev = &self.dev;
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?.get_ids().to_vec();
        let input_token_ids = Tensor::new(&tokens[..], dev)?.unsqueeze(0)?;

        let speaker_pcm = match self.speakers.get(speaker_index) {
            None => anyhow::bail!("unknown speaker id {speaker_index}"),
            Some(s) => &s.pcm,
        };
        let conditions = self.model.conditions(&input_token_ids, Some(speaker_pcm))?;
        tracing::info!("COND\n{conditions}");
        let audio_tokens = self.model.sample_lp(&conditions, cfg_alpha, lp)?;
        let audio = Tensor::new(audio_tokens, dev)?.unsqueeze(0)?.t()?;
        tracing::info!("sampled {:?}", audio.shape());
        let wav = self.codes_to_wav(&audio)?;
        tracing::info!("generated wav: {}", wav.len());
        Ok(wav)
    }

    fn codes_to_wav(&mut self, codes: &Tensor) -> Result<Vec<u8>> {
        use std::io::Write;

        let (_b_sz, _num_codebooks, steps) = codes.dims3()?;
        let nbins = self.mimi.config().quantizer_bins as u32;
        let pcm = {
            let mut pcms = vec![];
            self.mimi.reset_state();
            for step in 0..steps {
                let codes = codes.i((.., .., step..step + 1))?;
                let codes_v = codes.i((0, .., 0))?.to_vec1::<u32>()?;
                if codes_v.iter().any(|&v| v >= nbins) {
                    tracing::info!("encountered invalid bins {codes_v:?}, skipping {step}/{steps}");
                    break;
                }
                let codes = codes.into();
                let out = self.mimi.decode_step(&codes, &().into())?;
                if let Some(pcm) = out.as_option() {
                    let pcm = pcm.i((0, 0))?;
                    pcms.push(pcm.clone());
                }
            }
            Tensor::cat(&pcms, 0)?
        };
        tracing::info!("generated pcm: {:?}", pcm.shape());
        let pcm = pcm.to_vec1::<f32>()?;
        let mut out = Vec::new();
        moshi::wav::write_pcm_as_wav(&mut out, &pcm, 24000)?;
        let mut file = std::fs::File::create("/home/laurent/tmp/foo.wav")?;
        file.write_all(&out)?;
        Ok(out)
    }
}
