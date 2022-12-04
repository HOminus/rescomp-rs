use crate::ReservoirValue;
use nalgebra::{DVector, DVectorSlice};
use std::fmt::Debug;

pub trait ReservoirTimeEvolution<T: ReservoirValue>: Debug {
    fn input_dimension(&self) -> usize;

    fn output_dimension(&self) -> usize;

    fn time_evolution(&self, state: &mut DVector<T>, input: DVectorSlice<T>);
}

impl<T: ReservoirValue, E: ReservoirTimeEvolution<T>> ReservoirTimeEvolution<T> for Box<E> {
    fn input_dimension(&self) -> usize {
        (**self).input_dimension()
    }

    fn output_dimension(&self) -> usize {
        (**self).output_dimension()
    }

    fn time_evolution(&self, state: &mut DVector<T>, input: DVectorSlice<T>) {
        (**self).time_evolution(state, input);
    }
}
