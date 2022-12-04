use std::fmt::Debug;

use nalgebra::{DVector, DVectorSlice};

use crate::ReservoirValue;

pub trait ControlledReservoirTimeEvolution<T: ReservoirValue>: Debug {
    fn input_dimension(&self) -> usize;

    fn control_input_dimension(&self) -> usize;

    fn output_dimension(&self) -> usize;

    fn controlled_time_evolution(
        &self,
        state: &mut DVector<T>,
        input: DVectorSlice<T>,
        control: DVectorSlice<T>,
    );
}
