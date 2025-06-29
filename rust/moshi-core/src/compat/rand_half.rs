//! Compatibility module for half types with rand
//!
//! This module provides wrapper types that implement the necessary traits
//! for bf16 and f16 types from the half crate to work with rand's uniform distribution.
//! 
//! Since we can't directly implement foreign traits for foreign types due to Rust's orphan rules,
//! we use the newtype pattern with transparent wrappers.

use half::{bf16, f16};
use rand::distributions::uniform::{SampleUniform, SampleBorrow, UniformSampler};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::ops::{Deref, DerefMut};

/// A wrapper around f16 that implements SampleUniform
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct F16Wrapper(pub f16);

impl Deref for F16Wrapper {
    type Target = f16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for F16Wrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<f16> for F16Wrapper {
    fn from(v: f16) -> Self {
        F16Wrapper(v)
    }
}

impl From<F16Wrapper> for f16 {
    fn from(v: F16Wrapper) -> Self {
        v.0
    }
}

impl SampleUniform for F16Wrapper {
    type Sampler = UniformF16Wrapper;
}

/// Uniform sampler for F16Wrapper
#[derive(Clone, Copy, Debug)]
pub struct UniformF16Wrapper {
    low: F16Wrapper,
    range: F16Wrapper,
    scale: F16Wrapper,
    offset: F16Wrapper,
}

impl UniformSampler for UniformF16Wrapper {
    type X = F16Wrapper;
    
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X>,
        B2: SampleBorrow<Self::X>,
    {
        let low = *low.borrow();
        let high = *high.borrow();
        assert!(low.0 < high.0, "Uniform::new called with low >= high");
        
        let range = F16Wrapper(high.0 - low.0);
        
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
        assert!(low.0 <= high.0, "Uniform::new_inclusive called with low > high");
        
        let range = F16Wrapper(high.0 - low.0);
        
        let scale = range;
        let offset = low;
        
        Self {
            low,
            range,
            scale,
            offset,
        }
    }
    
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        // We use the same trick that rand uses internally for float sampling:
        // Generate a value in the range [0, 1) and scale to our target range
        let sampler = Uniform::new(0.0f32, 1.0f32);
        let f = Distribution::sample(&sampler, rng);
        
        // Scale to target range
        F16Wrapper(f16::from_f32(f) * self.scale.0 + self.offset.0)
    }
}

/// A wrapper around bf16 that implements SampleUniform
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct BF16Wrapper(pub bf16);

impl Deref for BF16Wrapper {
    type Target = bf16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BF16Wrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<bf16> for BF16Wrapper {
    fn from(v: bf16) -> Self {
        BF16Wrapper(v)
    }
}

impl From<BF16Wrapper> for bf16 {
    fn from(v: BF16Wrapper) -> Self {
        v.0
    }
}

impl SampleUniform for BF16Wrapper {
    type Sampler = UniformBF16Wrapper;
}

/// Uniform sampler for BF16Wrapper
#[derive(Clone, Copy, Debug)]
pub struct UniformBF16Wrapper {
    low: BF16Wrapper,
    range: BF16Wrapper,
    scale: BF16Wrapper,
    offset: BF16Wrapper,
}

impl UniformSampler for UniformBF16Wrapper {
    type X = BF16Wrapper;
    
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X>,
        B2: SampleBorrow<Self::X>,
    {
        let low = *low.borrow();
        let high = *high.borrow();
        assert!(low.0 < high.0, "Uniform::new called with low >= high");
        
        let range = BF16Wrapper(high.0 - low.0);
        
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
        assert!(low.0 <= high.0, "Uniform::new_inclusive called with low > high");
        
        let range = BF16Wrapper(high.0 - low.0);
        
        let scale = range;
        let offset = low;
        
        Self {
            low,
            range,
            scale,
            offset,
        }
    }
    
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        let sampler = Uniform::new(0.0f32, 1.0f32);
        let f = Distribution::sample(&sampler, rng);
        
        BF16Wrapper(bf16::from_f32(f) * self.scale.0 + self.offset.0)
    }
}

// Extension traits to make using with Uniform more ergonomic
pub trait UniformHalfExt {
    fn uniform_f16(low: f16, high: f16) -> Uniform<F16Wrapper>;
    fn uniform_bf16(low: bf16, high: bf16) -> Uniform<BF16Wrapper>;
}

impl UniformHalfExt for Uniform<f32> {
    fn uniform_f16(low: f16, high: f16) -> Uniform<F16Wrapper> {
        Uniform::new(F16Wrapper(low), F16Wrapper(high))
    }
    
    fn uniform_bf16(low: bf16, high: bf16) -> Uniform<BF16Wrapper> {
        Uniform::new(BF16Wrapper(low), BF16Wrapper(high))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_f16_uniform() {
        let seed = [42u8; 32];
        let mut rng = StdRng::from_seed(seed);
        
        let dist = Uniform::uniform_f16(f16::from_f32(0.0), f16::from_f32(1.0));
        let wrapper = dist.sample(&mut rng);
        let val: f16 = wrapper.into();
        
        assert!(val >= f16::from_f32(0.0));
        assert!(val < f16::from_f32(1.0));
    }

    #[test]
    fn test_bf16_uniform() {
        let seed = [42u8; 32];
        let mut rng = StdRng::from_seed(seed);
        
        let dist = Uniform::uniform_bf16(bf16::from_f32(0.0), bf16::from_f32(1.0));
        let wrapper = dist.sample(&mut rng);
        let val: bf16 = wrapper.into();
        
        assert!(val >= bf16::from_f32(0.0));
        assert!(val < bf16::from_f32(1.0));
    }
} 