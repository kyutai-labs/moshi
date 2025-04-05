//! Compatibility module for half types with rand
//!
//! This module implements the necessary traits for bf16 and f16 types
//! from the half crate to work with rand's uniform distribution.
//! 
//! The implementation follows the same pattern as for f32 and f64 in rand.

use half::{bf16, f16};
use rand::distributions::uniform::{SampleUniform, SampleBorrow, UniformSampler};
use rand::distributions::Uniform;
use std::ops::{Sub, Add};

/// Implementations for f16
impl SampleUniform for f16 {
    type Sampler = UniformF16;
}

/// Uniform sampler for f16
#[derive(Clone, Copy, Debug)]
pub struct UniformF16 {
    low: f16,
    range: f16,
    // These are used by the distribution to ensure the range is covered
    // properly and to enable distributions over ranges like 0..1.
    scale: f16,
    offset: f16,
}

impl UniformSampler for UniformF16 {
    type X = f16;
    
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X>,
        B2: SampleBorrow<Self::X>,
    {
        let low = *low.borrow();
        let high = *high.borrow();
        assert!(low < high, "Uniform::new called with low >= high");
        
        let range = high - low;
        
        // Calculate offset and scale used to map from the half-open range 
        // [0, 1) to the target range
        let scale = range;
        let offset = low;
        
        Self {
            low,
            range,
            scale,
            offset,
        }
    }
    
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X>,
        B2: SampleBorrow<Self::X>,
    {
        let low = *low.borrow();
        let high = *high.borrow();
        assert!(low <= high, "Uniform::new_inclusive called with low > high");
        
        let range = high - low;
        
        let scale = range;
        let offset = low;
        
        Self {
            low,
            range,
            scale,
            offset,
        }
    }
    
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        // We use the same trick that rand uses internally for float sampling:
        // Generate a value in the range [0, 1) and scale to our target range
        let sampler = Uniform::new(0.0f32, 1.0f32);
        let f = sampler.sample(rng);
        
        // Scale to target range
        f16::from_f32(f) * self.scale + self.offset
    }
}

/// Implementations for bf16
impl SampleUniform for bf16 {
    type Sampler = UniformBF16;
}

/// Uniform sampler for bf16
#[derive(Clone, Copy, Debug)]
pub struct UniformBF16 {
    low: bf16,
    range: bf16,
    scale: bf16,
    offset: bf16,
}

impl UniformSampler for UniformBF16 {
    type X = bf16;
    
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X>,
        B2: SampleBorrow<Self::X>,
    {
        let low = *low.borrow();
        let high = *high.borrow();
        assert!(low < high, "Uniform::new called with low >= high");
        
        let range = high - low;
        
        let scale = range;
        let offset = low;
        
        Self {
            low,
            range,
            scale,
            offset,
        }
    }
    
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X>,
        B2: SampleBorrow<Self::X>,
    {
        let low = *low.borrow();
        let high = *high.borrow();
        assert!(low <= high, "Uniform::new_inclusive called with low > high");
        
        let range = high - low;
        
        let scale = range;
        let offset = low;
        
        Self {
            low,
            range,
            scale,
            offset,
        }
    }
    
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        let sampler = Uniform::new(0.0f32, 1.0f32);
        let f = sampler.sample(rng);
        
        bf16::from_f32(f) * self.scale + self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Distribution;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_f16_uniform() {
        let seed = [42u8; 32];
        let mut rng = StdRng::from_seed(seed);
        
        let dist = Uniform::new(f16::from_f32(0.0), f16::from_f32(1.0));
        let val = dist.sample(&mut rng);
        
        assert!(val >= f16::from_f32(0.0));
        assert!(val < f16::from_f32(1.0));
    }

    #[test]
    fn test_bf16_uniform() {
        let seed = [42u8; 32];
        let mut rng = StdRng::from_seed(seed);
        
        let dist = Uniform::new(bf16::from_f32(0.0), bf16::from_f32(1.0));
        let val = dist.sample(&mut rng);
        
        assert!(val >= bf16::from_f32(0.0));
        assert!(val < bf16::from_f32(1.0));
    }
} 