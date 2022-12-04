use std::fmt::Debug;

use nalgebra::{
    base::{DMatrix, DMatrixSlice, DVector},
    DMatrixSliceMut, DVectorSliceMut,
};
use num_traits::Zero;

use super::ReservoirStateMeasurement;
use crate::ReservoirValue;

#[derive(Clone, Debug)]
pub struct DefaultStateMeasurement<T: ReservoirValue> {
    transformed_state: DVector<T>,
}

impl<T: ReservoirValue + Zero> DefaultStateMeasurement<T> {
    pub fn new(input_dimension: usize) -> Self {
        Self {
            transformed_state: DVector::<T>::zeros(input_dimension),
        }
    }
}

impl<T: ReservoirValue> ReservoirStateMeasurement<T> for DefaultStateMeasurement<T> {
    fn output_dimension(&self) -> usize {
        self.transformed_state.nrows()
    }

    fn measure(&mut self, data: &DVector<T>) -> &DVector<T> {
        self.transformed_state.copy_from(data);
        &self.transformed_state
    }

    fn measure_into(&self, state: &DVector<T>, mut target: DVectorSliceMut<T>) {
        target.copy_from(state);
    }

    fn measure_many(&self, data: DMatrixSlice<T>) -> DMatrix<T> {
        data.clone_owned()
    }

    fn measure_many_into(&self, states: DMatrixSlice<T>, mut targets: DMatrixSliceMut<T>) {
        targets.copy_from(&states)
    }
}

#[cfg(test)]
mod tests {
    use super::DefaultStateMeasurement;
    use crate::state_measurement::ReservoirStateMeasurement;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_extended_lu() {
        let mut default_measure = DefaultStateMeasurement::new(2);

        let state = DVector::from_vec(vec![1., 2.]);
        let measure_result = default_measure.measure(&state);
        assert_eq!(measure_result.as_slice(), &[1., 2.]);

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let states = DMatrix::from_vec(2, 5, data);
        let measure_results = default_measure.measure_many(states.columns(0, 5));
        assert_eq!(measure_results.column(0).as_slice(), &[1., 2.]);
        assert_eq!(measure_results.column(1).as_slice(), &[3., 4.]);
        assert_eq!(measure_results.column(2).as_slice(), &[5., 6.]);
        assert_eq!(measure_results.column(3).as_slice(), &[7., 8.]);
        assert_eq!(measure_results.column(4).as_slice(), &[9., 10.]);
    }
}
