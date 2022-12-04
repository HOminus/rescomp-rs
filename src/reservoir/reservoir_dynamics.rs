use std::{cmp::Ordering, fmt::Debug, marker::PhantomData};

use crate::input_projection::ReservoirInputProjection;
use crate::output_projection::ReservoirStateProjection;
use crate::state_measurement::ReservoirStateMeasurement;
use crate::time_evolution::ReservoirTimeEvolution;
use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, DVector};

use crate::ReservoirValue;

use super::Reservoir;

#[derive(Debug)]
pub struct ReservoirDynamics<T, I, E>
where
    T: ReservoirValue,
    E: ReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
{
    reservoir_input_projection: I,
    reservoir_time_evolution: E,
    _phantom: PhantomData<T>,
}

impl<T, I, E> Clone for ReservoirDynamics<T, I, E>
where
    T: ReservoirValue + Clone,
    I: ReservoirInputProjection<T> + Clone,
    E: ReservoirTimeEvolution<T> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            reservoir_input_projection: self.reservoir_input_projection.clone(),
            reservoir_time_evolution: self.reservoir_time_evolution.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T, I, E> ReservoirDynamics<T, I, E>
where
    T: ReservoirValue,
    E: ReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
{
    pub fn new(reservoir_input_projection: I, reservoir_time_evolution: E) -> Self {
        Self {
            reservoir_input_projection,
            reservoir_time_evolution,
            _phantom: PhantomData,
        }
    }

    pub fn into_parts(self) -> (I, E) {
        (
            self.reservoir_input_projection,
            self.reservoir_time_evolution,
        )
    }

    pub fn into_reservoir(self, state: DVector<T>) -> Reservoir<T, I, E> {
        Reservoir {
            reservoir_state: state,
            reservoir_dynamics: self,
        }
    }

    pub fn input_projection(&self) -> &I {
        &self.reservoir_input_projection
    }

    pub fn input_projection_mut(&mut self) -> &mut I {
        &mut self.reservoir_input_projection
    }

    pub fn time_evolution(&self) -> &E {
        &self.reservoir_time_evolution
    }

    pub fn synchronize_state(&mut self, state: &mut DVector<T>, input: DMatrixSlice<T>) {
        assert_eq!(input.nrows(), self.input_projection().input_dimension());
        let input_columns = self.input_projection().required_input_columns();
        let total_sync_steps = input.ncols() - input_columns + 1;

        for step in 0..total_sync_steps {
            let input = input.columns(step, input_columns);

            let input_vector = self.reservoir_input_projection.project(input);
            self.reservoir_time_evolution
                .time_evolution(state, input_vector.column(0));
        }
    }

    pub fn record_states(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
    ) -> DMatrix<T> {
        let mut states = DMatrix::zeros(
            self.time_evolution().output_dimension(),
            input.ncols() - sync_steps,
        );
        let slice = states.columns_mut(0, states.ncols());
        self.record_states_into(state, input, sync_steps, slice);
        states
    }

    pub fn record_states_into(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        mut result: DMatrixSliceMut<T>,
    ) {
        let data_points = input.ncols();
        let required_input_columns = self.reservoir_input_projection.required_input_columns();

        let synchronization_slice = input.columns(0, sync_steps);
        let train_slice = input.columns(
            sync_steps - required_input_columns,
            data_points - sync_steps,
        );
        self.synchronize_state(state, synchronization_slice);

        for step in 0..(data_points - sync_steps - required_input_columns + 1) {
            let current_train_slice = train_slice.columns(step, required_input_columns);

            let input_vector = self.reservoir_input_projection.project(current_train_slice);
            self.reservoir_time_evolution
                .time_evolution(state, input_vector.column(0));

            result.columns_mut(step, 1).copy_from(state);
        }
    }

    pub fn synchronize_and_predict<
        M: ReservoirStateMeasurement<T>,
        P: ReservoirStateProjection<T>,
    >(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        predict_steps: usize,
        measurement: &mut M,
        projection: &mut P,
    ) -> DMatrix<T> {
        let mut predictions = DMatrix::zeros(projection.output_dimension(), predict_steps);
        let slice = predictions.columns_mut(0, predict_steps);
        self.synchronize_and_predict_into(
            state,
            input,
            sync_steps,
            predict_steps,
            measurement,
            projection,
            slice,
        );
        predictions
    }

    #[allow(clippy::too_many_arguments)]
    pub fn synchronize_and_predict_into<
        M: ReservoirStateMeasurement<T>,
        P: ReservoirStateProjection<T>,
    >(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        predict_steps: usize,
        measurement: &mut M,
        projection: &mut P,
        mut result: DMatrixSliceMut<T>,
    ) {
        assert_eq!(
            sync_steps, 0,
            "Synchronization before prediction is not yet supported."
        );

        let input_columns = self.input_projection().required_input_columns();
        assert_eq!(
            input.ncols(),
            input_columns,
            "Only a single element as kickstarter is supported."
        );

        if input_columns > 1 {
            let mut overlapped_data = DMatrix::zeros(result.nrows(), 2 * input_columns);
            overlapped_data
                .columns_mut(0, input_columns - 1)
                .copy_from(&input.columns(1, input_columns - 1));

            let input = self.reservoir_input_projection.project(input);
            self.reservoir_time_evolution
                .time_evolution(state, input.column(0));
            for step in 0..usize::min(input_columns, predict_steps) {
                let state_measurement = measurement.measure(state);
                let prediction = projection.project(state_measurement);

                result.column_mut(step).copy_from(prediction);
                overlapped_data
                    .column_mut(input_columns + step - 1)
                    .copy_from(prediction);

                let input = self
                    .reservoir_input_projection
                    .project(overlapped_data.columns(step, input_columns));
                self.reservoir_time_evolution
                    .time_evolution(state, input.column(0));
            }

            for step in input_columns..predict_steps {
                let state_measurement = measurement.measure(state);
                let prediction = projection.project(state_measurement);

                result.column_mut(step).copy_from(prediction);

                let input = self
                    .reservoir_input_projection
                    .project(result.columns(step - input_columns, input_columns));
                self.reservoir_time_evolution
                    .time_evolution(state, input.column(0));
            }
        } else {
            let input = self.reservoir_input_projection.project(input);
            self.reservoir_time_evolution
                .time_evolution(state, input.column(0));
            for step in 0..predict_steps {
                let state_measurement = measurement.measure(state);
                let prediction = projection.project(state_measurement);

                result.column_mut(step).copy_from(prediction);

                let input = self
                    .reservoir_input_projection
                    .project(result.columns(step, 1));
                self.reservoir_time_evolution
                    .time_evolution(state, input.column(0));
            }
        }
    }

    pub fn predict_from_input_sequence<
        M: ReservoirStateMeasurement<T>,
        P: ReservoirStateProjection<T>,
    >(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        measurement: &mut M,
        projection: &mut P,
    ) -> DMatrix<T> {
        let predict_steps = input.ncols() - self.input_projection().required_input_columns() + 1;
        let mut predictions = DMatrix::zeros(projection.output_dimension(), predict_steps);
        let slice = predictions.columns_mut(0, predict_steps);
        self.predict_from_input_sequence_into(
            state,
            input,
            sync_steps,
            measurement,
            projection,
            slice,
        );
        predictions
    }

    pub fn predict_from_input_sequence_into<
        M: ReservoirStateMeasurement<T>,
        P: ReservoirStateProjection<T>,
    >(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        measurement: &mut M,
        projection: &mut P,
        mut result: DMatrixSliceMut<T>,
    ) {
        assert_eq!(
            sync_steps, 0,
            "Synchronization before prediction is not yet supported."
        );

        match input
            .ncols()
            .cmp(&self.input_projection().required_input_columns())
        {
            Ordering::Equal => {
                let input_projection = self.reservoir_input_projection.project(input);
                self.reservoir_time_evolution
                    .time_evolution(state, input_projection.column(0));
                let state_measurement = measurement.measure(state);
                projection.project_into(state_measurement, result.column_mut(0));
            }
            Ordering::Greater => {
                let input_projection = self.input_projection().project_many(input);

                let mut reservoir_states = DMatrix::zeros(state.nrows(), input_projection.ncols());
                for (index, input) in input_projection.column_iter().enumerate() {
                    self.time_evolution().time_evolution(state, input);
                    reservoir_states.column_mut(index).copy_from(state);
                }

                let state_measurements =
                    measurement.measure_many(reservoir_states.columns(0, reservoir_states.ncols()));
                projection.project_many_into(
                    state_measurements.columns(0, state_measurements.ncols()),
                    result,
                );
            }
            _ => panic!(),
        }
    }
}
