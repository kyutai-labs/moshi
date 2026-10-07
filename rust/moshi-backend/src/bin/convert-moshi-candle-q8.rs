// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Convert a PyTorch Moshi safetensors checkpoint to the GGUF layout expected by
//! the Candle backend and quantize its weights to Q8_0.

use anyhow::{bail, Context, Result};
use candle::{
    quantized::{gguf_file, GgmlDType, QTensor},
    safetensors::MmapedSafetensors,
    Device,
};
use clap::Parser;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Parser)]
struct Args {
    /// PyTorch Moshi checkpoint. LoRA adapters must be merged beforehand.
    #[arg(long)]
    input: PathBuf,

    /// Destination Candle-compatible Q8_0 GGUF file.
    #[arg(long)]
    output: PathBuf,

    /// Validate all tensor mappings and shapes without quantizing.
    #[arg(long)]
    validate_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TargetTensor {
    name: String,
    /// `(dimension, offset, length)` to select from the source tensor.
    narrow: Option<(usize, usize, usize)>,
}

fn indexed_name(name: &str, prefix: &str, suffix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.strip_suffix(suffix)?.parse().ok()
}

fn depformer_slices(source_names: &[String]) -> Result<usize> {
    let mut indices = source_names
        .iter()
        .filter_map(|name| indexed_name(name, "depformer_in.", ".weight"))
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() || indices.iter().copied().ne(0..indices.len()) {
        bail!("depformer_in indices must be contiguous and start at zero, got {indices:?}")
    }
    Ok(indices.len())
}

fn target_tensors(name: &str, shape: &[usize], num_slices: usize) -> Result<Vec<TargetTensor>> {
    let one = |name: String| vec![TargetTensor { name, narrow: None }];

    if name == "depformer_text_emb.weight" {
        return Ok(one("depformer.0.emb.weight".into()));
    }
    if let Some(index) = indexed_name(name, "depformer_emb.", ".weight") {
        if index + 1 >= num_slices {
            bail!("invalid depformer embedding index {index} for {num_slices} slices")
        }
        return Ok(one(format!("depformer.{}.emb.weight", index + 1)));
    }
    if let Some(index) = indexed_name(name, "depformer_in.", ".weight") {
        return Ok(one(format!("depformer.{index}.linear_in.weight")));
    }
    if let Some(index) = indexed_name(name, "linears.", ".weight") {
        if index >= num_slices {
            bail!("invalid depformer output index {index} for {num_slices} slices")
        }
        return Ok(one(format!("depformer.{index}.linear_out.weight")));
    }

    let parts = name.split('.').collect::<Vec<_>>();
    match parts.as_slice() {
        ["depformer", "layers", layer, "gating", slice, rest @ ..] => {
            let slice: usize = slice.parse()?;
            if slice >= num_slices {
                bail!("invalid depformer gating slice {slice} for {num_slices} slices")
            }
            Ok(one(format!(
                "depformer.{slice}.transformer.layers.{layer}.gating.{}",
                rest.join(".")
            )))
        }
        ["depformer", "layers", layer, norm @ ("norm1" | "norm2"), "alpha"] => Ok((0..num_slices)
            .map(|slice| TargetTensor {
                name: format!("depformer.{slice}.transformer.layers.{layer}.{norm}.alpha"),
                narrow: None,
            })
            .collect()),
        ["depformer", "layers", layer, "self_attn", rest @ ..] => {
            let first_dim = *shape.first().context("depformer attention tensor is a scalar")?;
            if first_dim % num_slices != 0 {
                bail!("first dimension of {name} ({first_dim}) is not divisible by {num_slices}")
            }
            let slice_len = first_dim / num_slices;
            Ok((0..num_slices)
                .map(|slice| TargetTensor {
                    name: format!(
                        "depformer.{slice}.transformer.layers.{layer}.self_attn.{}",
                        rest.join(".")
                    ),
                    narrow: Some((0, slice * slice_len, slice_len)),
                })
                .collect())
        }
        ["depformer", ..] => bail!("unsupported depformer tensor {name}"),
        _ => Ok(one(name.into())),
    }
}

fn quantization_type(name: &str) -> GgmlDType {
    if name.ends_with(".alpha") {
        GgmlDType::F32
    } else {
        GgmlDType::Q8_0
    }
}

fn temporary_path(output: &Path) -> Result<PathBuf> {
    let file_name = output.file_name().and_then(|v| v.to_str()).context("invalid output path")?;
    Ok(output.with_file_name(format!(".{file_name}.tmp")))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let source = unsafe { MmapedSafetensors::new(&args.input)? };
    let mut source_infos = source
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect::<Vec<_>>();
    source_infos.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let source_names = source_infos.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
    let num_slices = depformer_slices(&source_names)?;

    let mut mapped = Vec::new();
    for (source_name, source_shape) in &source_infos {
        for target in target_tensors(source_name, source_shape, num_slices)? {
            mapped.push((target.name, source_name.clone(), target.narrow));
        }
    }
    mapped.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for pair in mapped.windows(2) {
        if pair[0].0 == pair[1].0 {
            bail!("multiple source tensors map to {}", pair[0].0)
        }
    }

    let mut names = Vec::with_capacity(mapped.len());
    let mut tensors = Vec::with_capacity(mapped.len());
    for (index, (target_name, source_name, narrow)) in mapped.iter().enumerate() {
        let mut tensor = source
            .load(source_name, &Device::Cpu)
            .with_context(|| format!("loading {source_name}"))?;
        if let Some((dimension, offset, length)) = narrow {
            tensor = tensor.narrow(*dimension, *offset, *length)?;
        }
        let dtype = quantization_type(target_name);
        eprintln!(
            "[{}/{}] {source_name} -> {target_name} {:?} {dtype:?}",
            index + 1,
            mapped.len(),
            tensor.dims(),
        );
        if !args.validate_only {
            tensors.push(
                QTensor::quantize(&tensor, dtype)
                    .with_context(|| format!("quantizing {target_name}"))?,
            );
            names.push(target_name.clone());
        }
    }
    eprintln!("mapped {} PyTorch tensors to {} Candle tensors", source_infos.len(), mapped.len());

    if args.validate_only {
        return Ok(());
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(&args.output)?;
    let pairs = names.iter().zip(tensors.iter()).map(|(name, tensor)| (name.as_str(), tensor));
    let pairs = pairs.collect::<Vec<_>>();
    let mut writer = BufWriter::new(File::create(&temporary)?);
    gguf_file::write(&mut writer, &[], &pairs)
        .with_context(|| format!("writing {}", temporary.display()))?;
    writer.flush()?;
    drop(writer);
    std::fs::rename(&temporary, &args.output)?;
    eprintln!("wrote {} tensors to {}", pairs.len(), args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_depformer_tensors_to_candle_layout() -> Result<()> {
        assert_eq!(
            target_tensors("depformer_text_emb.weight", &[32001, 1024], 8)?,
            vec![TargetTensor { name: "depformer.0.emb.weight".into(), narrow: None }]
        );
        assert_eq!(
            target_tensors("depformer_emb.0.weight", &[2049, 1024], 8)?,
            vec![TargetTensor { name: "depformer.1.emb.weight".into(), narrow: None }]
        );
        assert_eq!(
            target_tensors("depformer.layers.2.gating.3.linear_in.weight", &[5632, 1024], 8)?,
            vec![TargetTensor {
                name: "depformer.3.transformer.layers.2.gating.linear_in.weight".into(),
                narrow: None,
            }]
        );
        let attention =
            target_tensors("depformer.layers.2.self_attn.in_proj_weight", &[24576, 1024], 8)?;
        assert_eq!(attention.len(), 8);
        assert_eq!(attention[3].narrow, Some((0, 9216, 3072)));
        assert_eq!(attention[3].name, "depformer.3.transformer.layers.2.self_attn.in_proj_weight");
        Ok(())
    }

    #[test]
    fn maps_shared_norm_to_every_depformer_slice() -> Result<()> {
        let tensors = target_tensors("depformer.layers.5.norm1.alpha", &[1, 1, 1024], 8)?;
        assert_eq!(tensors.len(), 8);
        assert_eq!(tensors[7].name, "depformer.7.transformer.layers.5.norm1.alpha");
        assert!(tensors.iter().all(|tensor| tensor.narrow.is_none()));
        Ok(())
    }

    #[test]
    fn preserves_main_transformer_names() -> Result<()> {
        let tensors =
            target_tensors("transformer.layers.0.self_attn.in_proj_weight", &[12288, 4096], 8)?;
        assert_eq!(
            tensors,
            vec![TargetTensor {
                name: "transformer.layers.0.self_attn.in_proj_weight".into(),
                narrow: None,
            }]
        );
        Ok(())
    }
}
