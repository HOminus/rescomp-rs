use nalgebra::{ClosedAdd, ClosedMul, ComplexField, DMatrix, DMatrixSlice};
use num_traits::Float;

use crate::{
    input_projection::ReservoirInputProjection, output_projection::LinearStateProjection,
    state_measurement::ReservoirStateMeasurement, time_evolution::ReservoirTimeEvolution,
    Reservoir, ReservoirComputer, ReservoirValue,
};

pub struct ReservoirTraining<T>
where
    T: ReservoirValue + ComplexField + ClosedAdd + ClosedMul,
{
    data: Vec<DMatrix<T>>,
    train_sync_steps: usize,
    train_steps: usize,
    prediction_sync_steps: usize,
    prediction_steps: usize,
}

impl<T> ReservoirTraining<T>
where
    T: ReservoirValue + ComplexField + ClosedAdd + ClosedMul,
{
    pub fn new(
        train_sync_steps: usize,
        train_steps: usize,
        prediction_sync_steps: usize,
        prediction_steps: usize,
    ) -> Self {
        Self {
            data: vec![],
            train_sync_steps,
            train_steps,
            prediction_sync_steps,
            prediction_steps,
        }
    }

    pub fn add_data(&mut self, data: DMatrix<T>) -> &mut Self {
        assert!(
            data.ncols()
                >= self.train_sync_steps
                    + self.train_steps
                    + self.prediction_sync_steps
                    + self.prediction_steps
        );
        self.data.push(data);
        self
    }

    pub fn train_via_ridge_regression<I, E, M>(
        &self,
        mut reservoir: Reservoir<T, I, E>,
        measurement: M,
    ) -> ReservoirComputer<T, I, E, M, LinearStateProjection<T>>
    where
        I: ReservoirInputProjection<T>,
        E: ReservoirTimeEvolution<T>,
        M: ReservoirStateMeasurement<T>,
    {
        assert_eq!(
            self.data.len(),
            1,
            "Multifunctional learning scenarios are not yet supported."
        );
        let data = &self.data[0];

        let sync_train_steps = self.train_sync_steps + self.train_steps;

        let sync_train_data = data.columns(0, sync_train_steps - 1);
        let recorded_states = reservoir.record_states(sync_train_data, self.train_sync_steps);
        let matching_data_states = data.columns(
            self.train_sync_steps + 1,
            sync_train_steps - self.train_sync_steps - 1,
        );

        let recorded_states =
            measurement.measure_many(recorded_states.columns(0, recorded_states.ncols()));
        let linear_fit = LinearStateProjection::via_ridge_regression_nalgebra(
            Float::powi(T::one(), -7),
            &recorded_states,
            matching_data_states,
        );

        ReservoirComputer {
            reservoir,
            reservoir_state_measurement: measurement,
            reservoir_state_projection: linear_fit,
        }
    }

    pub fn train_via_tikhonov_regularization<I, E, M>(
        &self,
        tikhonov: &DMatrix<T>,
        mut reservoir: Reservoir<T, I, E>,
        measurement: M,
    ) -> ReservoirComputer<T, I, E, M, LinearStateProjection<T>>
    where
        I: ReservoirInputProjection<T>,
        E: ReservoirTimeEvolution<T>,
        M: ReservoirStateMeasurement<T>,
    {
        assert_eq!(
            self.data.len(),
            1,
            "Multifunctional learning scenarios are not yet supported."
        );
        let data = &self.data[0];

        let sync_train_steps = self.train_sync_steps + self.train_steps;

        let sync_train_data = data.columns(0, sync_train_steps - 1);
        let recorded_states = reservoir.record_states(sync_train_data, self.train_sync_steps);
        let matching_data_states = data.columns(
            self.train_sync_steps + 1,
            sync_train_steps - self.train_sync_steps - 1,
        );

        let recorded_states =
            measurement.measure_many(recorded_states.columns(0, recorded_states.ncols()));
        let linear_fit = LinearStateProjection::via_tikhonov_regularization_nalgebra(
            tikhonov,
            &recorded_states,
            matching_data_states,
        );

        ReservoirComputer {
            reservoir,
            reservoir_state_measurement: measurement,
            reservoir_state_projection: linear_fit,
        }
    }

    pub fn get_prediction_kickstarter(
        &self,
        index: usize,
        required_elements: usize,
    ) -> DMatrixSlice<T> {
        self.data[index].columns(
            self.train_sync_steps + self.train_steps - required_elements,
            required_elements,
        )
    }

    pub fn get_true_future(&self, index: usize) -> DMatrixSlice<T> {
        let column_count = self.data[index].ncols();
        let start_offset = self.train_sync_steps + self.train_steps + self.prediction_sync_steps;
        let remaining = column_count - start_offset;
        self.data[index].columns(start_offset, remaining)
    }
}
