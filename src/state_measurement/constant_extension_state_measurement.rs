use std::fmt::Debug;

use nalgebra::{
    base::{DMatrix, DMatrixSlice, DVector},
    DMatrixSliceMut, DVectorSliceMut,
};

use super::ReservoirStateMeasurement;
use crate::ReservoirValue;

#[derive(Clone, Debug)]
pub struct ConstantExtensionStateMeasurement<T: ReservoirValue> {
    transformed_state: DVector<T>,
    const_val: T,
}

impl<T: ReservoirValue> ConstantExtensionStateMeasurement<T> {
    pub fn new(input_dimension: usize) -> Self {
        Self {
            transformed_state: DVector::<T>::zeros(input_dimension + 1),
            const_val: T::one(),
        }
    }

    fn impl_measure(state: &DVector<T>, val: T, mut target: DVectorSliceMut<T>) {
        target.rows_mut(0, state.nrows()).copy_from(state);
        target[state.nrows()] = val
    }

    fn impl_measure_many(states: DMatrixSlice<T>, val: T, mut targets: DMatrixSliceMut<T>) {
        assert_eq!(states.nrows() + 1, targets.nrows());
        assert_eq!(states.ncols(), targets.ncols());
        let state_dim = states.nrows();
        for (state_column, mut target_column) in states.column_iter().zip(targets.column_iter_mut())
        {
            target_column
                .rows_mut(0, state_dim)
                .copy_from(&state_column);
            target_column[state_dim] = val;
        }
    }
}

impl<T: ReservoirValue> ReservoirStateMeasurement<T> for ConstantExtensionStateMeasurement<T> {
    fn output_dimension(&self) -> usize {
        self.transformed_state.nrows()
    }

    fn measure(&mut self, state: &DVector<T>) -> &DVector<T> {
        Self::impl_measure(state, self.const_val, self.transformed_state.column_mut(0));
        &self.transformed_state
    }

    fn measure_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        Self::impl_measure(state, self.const_val, target)
    }

    fn measure_many(&self, states: DMatrixSlice<T>) -> DMatrix<T> {
        let mut targets = DMatrix::zeros(states.nrows() + 1, states.ncols());
        Self::impl_measure_many(
            states,
            self.const_val,
            targets.columns_mut(0, targets.ncols()),
        );
        targets
    }

    fn measure_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        Self::impl_measure_many(states, self.const_val, targets);
    }
}

#[cfg(test)]
mod tests {
    use super::ConstantExtensionStateMeasurement;
    use crate::state_measurement::ReservoirStateMeasurement;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_extended_lu() {
        let mut cnst_ext_measure = ConstantExtensionStateMeasurement::new(2);

        let state = DVector::from_vec(vec![1., 2.]);
        let measure_result = cnst_ext_measure.measure(&state);
        assert_eq!(measure_result.as_slice(), &[1., 2., 1.]);

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let states = DMatrix::from_vec(2, 5, data);
        let measure_results = cnst_ext_measure.measure_many(states.columns(0, 5));
        assert_eq!(measure_results.column(0).as_slice(), &[1., 2., 1.]);
        assert_eq!(measure_results.column(1).as_slice(), &[3., 4., 1.]);
        assert_eq!(measure_results.column(2).as_slice(), &[5., 6., 1.]);
        assert_eq!(measure_results.column(3).as_slice(), &[7., 8., 1.]);
        assert_eq!(measure_results.column(4).as_slice(), &[9., 10., 1.]);
    }
}
