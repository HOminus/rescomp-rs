use std::fmt::Debug;

use nalgebra::{
    base::{DMatrix, DMatrixSlice, DVector},
    DMatrixSliceMut, DVectorSliceMut,
};

use super::ReservoirStateMeasurement;
use crate::ReservoirValue;

#[derive(Clone, Debug)]
pub struct LuStateMeasurement<T: ReservoirValue> {
    transformed_state: DVector<T>,
}

impl<T: ReservoirValue> LuStateMeasurement<T> {
    pub fn new(input_dimension: usize) -> Self {
        Self {
            transformed_state: DVector::<T>::zeros(input_dimension),
        }
    }

    fn impl_measure(state: &DVector<T>, mut target: DVectorSliceMut<T>) {
        for (index, value) in state.iter().enumerate() {
            if index % 2 == 1 {
                target[index] = *value * *value;
            } else {
                target[index] = *value;
            }
        }
    }

    fn impl_measure_many(state: DMatrixSlice<T>, mut targets: DMatrixSliceMut<T>) {
        assert_eq!(state.nrows(), targets.nrows());
        assert_eq!(state.ncols(), targets.ncols());

        for (state_column, mut target_column) in state.column_iter().zip(targets.column_iter_mut())
        {
            for (index, value) in state_column.iter().enumerate() {
                if index % 2 == 1 {
                    target_column[index] = *value * *value;
                } else {
                    target_column[index] = *value;
                }
            }
        }
    }
}

impl<T: ReservoirValue> ReservoirStateMeasurement<T> for LuStateMeasurement<T> {
    fn output_dimension(&self) -> usize {
        self.transformed_state.nrows()
    }

    fn measure(&mut self, state: &DVector<T>) -> &DVector<T> {
        Self::impl_measure(state, self.transformed_state.column_mut(0));
        &self.transformed_state
    }

    fn measure_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        Self::impl_measure(state, target)
    }

    fn measure_many(&self, states: DMatrixSlice<T>) -> DMatrix<T> {
        let mut targets = DMatrix::zeros(states.nrows(), states.ncols());
        Self::impl_measure_many(states, targets.columns_mut(0, targets.ncols()));
        targets
    }

    fn measure_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        Self::impl_measure_many(states, targets);
    }
}

#[cfg(test)]
mod tests {
    use super::LuStateMeasurement;
    use crate::state_measurement::ReservoirStateMeasurement;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_extended_lu() {
        let mut lu_measure = LuStateMeasurement::new(2);

        let state = DVector::from_vec(vec![1., 2.]);
        let measure_result = lu_measure.measure(&state);
        assert_eq!(measure_result.as_slice(), &[1., 4.]);

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let states = DMatrix::from_vec(2, 5, data);
        let measure_results = lu_measure.measure_many(states.columns(0, 5));
        assert_eq!(measure_results.column(0).as_slice(), &[1., 4.]);
        assert_eq!(measure_results.column(1).as_slice(), &[3., 16.]);
        assert_eq!(measure_results.column(2).as_slice(), &[5., 36.]);
        assert_eq!(measure_results.column(3).as_slice(), &[7., 64.]);
        assert_eq!(measure_results.column(4).as_slice(), &[9., 100.]);
    }
}
