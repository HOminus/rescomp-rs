use std::marker::PhantomData;

use crate::ReservoirValue;

pub mod biased_activation_function;
pub use biased_activation_function::BiasedActivationFunction;

pub trait ActiviationFunction<T: ReservoirValue> {
    fn invoke(&self, index: usize, value: T) -> T;
}

impl<T: ReservoirValue, A: ActiviationFunction<T>> ActiviationFunction<T> for Box<A> {
    #[inline]
    fn invoke(&self, index: usize, value: T) -> T {
        (**self).invoke(index, value)
    }
}

impl<T: ReservoirValue> ActiviationFunction<T> for Box<dyn ActiviationFunction<T>> {
    fn invoke(&self, index: usize, value: T) -> T {
        (**self).invoke(index, value)
    }
}

pub struct ActivationFunctionWrapper<T: ReservoirValue, F: Fn(usize, T) -> T> {
    func: F,
    _phantom: PhantomData<T>,
}

impl<T: ReservoirValue, F: Fn(usize, T) -> T> ActivationFunctionWrapper<T, F> {
    pub fn new(f: F) -> Self {
        Self {
            func: f,
            _phantom: PhantomData,
        }
    }
}

impl<T: ReservoirValue, F: Fn(usize, T) -> T> ActiviationFunction<T>
    for ActivationFunctionWrapper<T, F>
{
    #[inline]
    fn invoke(&self, index: usize, value: T) -> T {
        (self.func)(index, value)
    }
}

impl<T: ReservoirValue, F: Fn(usize, T) -> T + Clone> Clone for ActivationFunctionWrapper<T, F> {
    fn clone(&self) -> Self {
        Self {
            func: self.func.clone(),
            _phantom: PhantomData,
        }
    }
}
