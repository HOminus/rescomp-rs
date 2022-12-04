use std::fmt::Debug;

use nalgebra::{
    base::{DMatrix, DMatrixSlice, DVector},
    DMatrixSliceMut, DVectorSliceMut,
};

use super::ReservoirStateMeasurement;
use crate::ReservoirValue;

#[derive(Debug, Clone)]
pub struct ExtendedLuStateMeasurement<T: ReservoirValue> {
    transformed_state: DVector<T>,
}

impl<T: ReservoirValue> ExtendedLuStateMeasurement<T> {
    pub fn new(input_dimensions: usize) -> Self {
        Self {
            transformed_state: DVector::<T>::zeros(2 * input_dimensions),
        }
    }

    fn impl_measure(state: &DVector<T>, mut target: DVectorSliceMut<T>) {
        let core_dimension = state.nrows();
        for (index, e) in state.as_slice().iter().enumerate() {
            target[index] = *e;
            target[core_dimension + index] = (*e) * (*e);
        }
    }

    fn impl_measure_many(state: DMatrixSlice<T>, mut targets: DMatrixSliceMut<T>) {
        let core_dimension = state.nrows();
        for (col_index, state_slice) in state.column_iter().enumerate() {
            let mut target_slice = targets.columns_mut(col_index, 1);

            for (row_index, e) in state_slice.as_slice().iter().enumerate() {
                target_slice[row_index] = *e;
                target_slice[core_dimension + row_index] = (*e) * (*e);
            }
        }
    }
}

impl<T: Copy + ReservoirValue> ReservoirStateMeasurement<T> for ExtendedLuStateMeasurement<T> {
    fn output_dimension(&self) -> usize {
        self.transformed_state.nrows()
    }

    fn measure(&mut self, state: &DVector<T>) -> &DVector<T> {
        assert_eq!(self.output_dimension(), 2 * state.nrows());
        let output_dimension = self.output_dimension();
        let slice =
            DVectorSliceMut::from_slice(self.transformed_state.as_mut_slice(), output_dimension);
        Self::impl_measure(state, slice);
        &self.transformed_state
    }

    fn measure_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        assert_eq!(self.output_dimension(), 2 * state.nrows());
        assert_eq!(target.nrows(), self.output_dimension());
        Self::impl_measure(state, target);
    }

    fn measure_many(&self, state: DMatrixSlice<T>) -> DMatrix<T> {
        assert_eq!(self.output_dimension(), 2 * state.nrows());
        let mut target = DMatrix::zeros(2 * state.nrows(), state.ncols());
        let slice = target.columns_mut(0, state.ncols());
        Self::impl_measure_many(state, slice);
        target
    }

    fn measure_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        assert_eq!(self.output_dimension(), 2 * targets.nrows());
        assert_eq!(targets.nrows(), self.output_dimension());
        assert_eq!(targets.ncols(), states.ncols());
        Self::impl_measure_many(states, targets);
    }
}

#[cfg(test)]
mod tests {
    use super::ExtendedLuStateMeasurement;
    use crate::state_measurement::ReservoirStateMeasurement;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_extended_lu() {
        let mut ex_lu_measure = ExtendedLuStateMeasurement::new(2);

        let state = DVector::from_vec(vec![1., 2.]);
        let measure_result = ex_lu_measure.measure(&state);
        assert_eq!(measure_result.as_slice(), &[1., 2., 1., 4.]);

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let states = DMatrix::from_vec(2, 5, data);
        let measure_results = ex_lu_measure.measure_many(states.columns(0, 5));
        assert_eq!(measure_results.column(0).as_slice(), &[1., 2., 1., 4.]);
        assert_eq!(measure_results.column(1).as_slice(), &[3., 4., 9., 16.]);
        assert_eq!(measure_results.column(2).as_slice(), &[5., 6., 25., 36.]);
        assert_eq!(measure_results.column(3).as_slice(), &[7., 8., 49., 64.]);
        assert_eq!(measure_results.column(4).as_slice(), &[9., 10., 81., 100.]);
    }
}
