pub use candle;
pub use candle_nn;

pub mod conv;
pub mod encodec;
pub mod lm;
pub mod lm_generate;
pub mod lm_generate_multistream;
pub mod quantization;
pub mod quantized_lm;
pub mod quantized_transformer;
pub mod seanet;
pub mod streaming;
pub mod transformer;
pub mod tts;
pub mod wav;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}
