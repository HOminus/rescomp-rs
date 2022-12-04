use std::fmt::Debug;

use nalgebra::DVector;
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    prelude::Distribution,
    thread_rng,
};

use crate::ReservoirValue;

use super::ActiviationFunction;

#[derive(Clone, Debug)]
pub struct BiasedActivationFunction<
    T: ReservoirValue + SampleUniform,
    F: Fn(T, T) -> T + Clone + Debug,
> {
    bias_scale: T,
    bias: DVector<T>,
    function: F,
}

impl<T: ReservoirValue + SampleUniform, F: Fn(T, T) -> T + Clone + Debug>
    BiasedActivationFunction<T, F>
{
    pub fn new_uniform_random(size: usize, f: F, scale: T) -> Self {
        let mut bias = DVector::zeros(size);

        let plus_minus_one = Uniform::new_inclusive(-T::one(), T::one());
        let mut rng = thread_rng();
        for e in bias.iter_mut() {
            *e = scale * plus_minus_one.sample(&mut rng);
        }

        Self {
            bias_scale: scale,
            function: f,
            bias,
        }
    }

    pub fn bias_scale(&self) -> T {
        self.bias_scale
    }
}

impl<T: ReservoirValue + SampleUniform, F: Fn(T, T) -> T + Clone + Debug> ActiviationFunction<T>
    for BiasedActivationFunction<T, F>
{
    fn invoke(&self, index: usize, value: T) -> T {
        (self.function)(self.bias[index], value)
    }
}
