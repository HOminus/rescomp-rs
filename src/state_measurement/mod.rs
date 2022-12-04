use std::fmt::Debug;

use nalgebra::{
    base::{DMatrix, DMatrixSlice, DVector},
    DMatrixSliceMut, DVectorSliceMut,
};

use crate::ReservoirValue;

pub mod constant_extension_state_measurement;
pub mod default_state_measurement;
pub mod extended_lu_state_measurement;
pub mod lu_state_measurement;

pub use constant_extension_state_measurement::ConstantExtensionStateMeasurement;
pub use default_state_measurement::DefaultStateMeasurement;
pub use extended_lu_state_measurement::ExtendedLuStateMeasurement;
pub use lu_state_measurement::LuStateMeasurement;

pub trait ReservoirStateMeasurement<T: ReservoirValue>: Debug {
    fn output_dimension(&self) -> usize;

    fn measure(&mut self, state: &DVector<T>) -> &DVector<T>;

    fn measure_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>);

    fn measure_many(&self, state: DMatrixSlice<T>) -> DMatrix<T>;

    fn measure_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>);
}

impl<T: ReservoirValue, M: ReservoirStateMeasurement<T>> ReservoirStateMeasurement<T> for Box<M> {
    fn output_dimension(&self) -> usize {
        (**self).output_dimension()
    }

    fn measure(&mut self, state: &DVector<T>) -> &DVector<T> {
        (**self).measure(state)
    }

    fn measure_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        (**self).measure_into(state, target);
    }

    fn measure_many(&self, state: DMatrixSlice<T>) -> DMatrix<T> {
        (**self).measure_many(state)
    }

    fn measure_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        (**self).measure_many_into(states, targets);
    }
}

impl<T: ReservoirValue> ReservoirStateMeasurement<T> for Box<dyn ReservoirStateMeasurement<T>> {
    fn output_dimension(&self) -> usize {
        (**self).output_dimension()
    }

    fn measure(&mut self, state: &DVector<T>) -> &DVector<T> {
        (**self).measure(state)
    }

    fn measure_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        (**self).measure_into(state, target);
    }

    fn measure_many(&self, state: DMatrixSlice<T>) -> DMatrix<T> {
        (**self).measure_many(state)
    }

    fn measure_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        (**self).measure_many_into(states, targets);
    }
}

/*
#[derive(Clone)]
pub struct PcaStateReadout<T: Copy + Scalar + Float + Field> {
    pca_components: usize,
    average_reservoir_state: DVector<T>,
    pca_matrix: DMatrix<T>,
    transformed_state: DVector<T>,
}
*/
