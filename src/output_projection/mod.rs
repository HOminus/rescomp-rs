use std::fmt::Debug;

use crate::ReservoirValue;
use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, DVector, DVectorSliceMut};

pub mod linear_state_projection;
pub use linear_state_projection::LinearStateProjection;

pub trait ReservoirStateProjection<T: ReservoirValue>: Debug {
    fn output_dimension(&self) -> usize;

    fn input_dimension(&self) -> usize;

    fn project(&mut self, state: &DVector<T>) -> &DVector<T>;

    fn project_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>);

    fn project_many(&self, states: DMatrixSlice<T>) -> DMatrix<T>;

    fn project_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>);
}

impl<T: ReservoirValue, P: ReservoirStateProjection<T>> ReservoirStateProjection<T> for Box<P> {
    fn output_dimension(&self) -> usize {
        (**self).output_dimension()
    }

    fn input_dimension(&self) -> usize {
        (**self).input_dimension()
    }

    fn project(&mut self, state: &DVector<T>) -> &DVector<T> {
        (**self).project(state)
    }

    fn project_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        (**self).project_into(state, target);
    }

    fn project_many(&self, states: DMatrixSlice<T>) -> DMatrix<T> {
        (**self).project_many(states)
    }

    fn project_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        (**self).project_many_into(states, targets);
    }
}

impl<T: ReservoirValue> ReservoirStateProjection<T> for Box<dyn ReservoirStateProjection<T>> {
    fn output_dimension(&self) -> usize {
        (**self).output_dimension()
    }

    fn input_dimension(&self) -> usize {
        (**self).input_dimension()
    }

    fn project(&mut self, state: &DVector<T>) -> &DVector<T> {
        (**self).project(state)
    }

    fn project_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        (**self).project_into(state, target);
    }

    fn project_many(&self, states: DMatrixSlice<T>) -> DMatrix<T> {
        (**self).project_many(states)
    }

    fn project_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        (**self).project_many_into(states, targets);
    }
}
