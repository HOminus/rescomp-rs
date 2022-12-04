use std::fmt::Debug;

use crate::input_projection::ReservoirInputProjection;
use crate::output_projection::ReservoirStateProjection;
use crate::state_measurement::ReservoirStateMeasurement;
use crate::time_evolution::ReservoirTimeEvolution;
use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, DVector};

use super::ReservoirDynamics;
use crate::ReservoirValue;

#[derive(Debug)]
pub struct Reservoir<T, I, E>
where
    T: ReservoirValue,
    E: ReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
{
    pub(crate) reservoir_state: DVector<T>,
    pub(crate) reservoir_dynamics: ReservoirDynamics<T, I, E>,
}

impl<T, I, E> Reservoir<T, I, E>
where
    T: ReservoirValue,
    E: ReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
{
    pub fn new(reservoir_input_projection: I, reservoir_time_evolution: E) -> Self {
        let reservoir_dimension = reservoir_time_evolution.input_dimension();
        let reservoir_dynamics =
            ReservoirDynamics::new(reservoir_input_projection, reservoir_time_evolution);
        Self {
            reservoir_dynamics,
            reservoir_state: DVector::zeros(reservoir_dimension),
        }
    }

    pub fn into_parts(self) -> (DVector<T>, I, E) {
        let (i, e) = self.reservoir_dynamics.into_parts();
        (self.reservoir_state, i, e)
    }

    pub fn split_reservoir_dynamics(self) -> (DVector<T>, ReservoirDynamics<T, I, E>) {
        (self.reservoir_state, self.reservoir_dynamics)
    }

    pub fn input_projection(&self) -> &I {
        self.reservoir_dynamics.input_projection()
    }

    pub fn time_evolution(&self) -> &E {
        self.reservoir_dynamics.time_evolution()
    }

    pub fn state(&self) -> &DVector<T> {
        &self.reservoir_state
    }

    pub fn synchronize_state(&mut self, input: DMatrixSlice<T>) {
        self.reservoir_dynamics
            .synchronize_state(&mut self.reservoir_state, input);
    }

    pub fn record_states(&mut self, input: DMatrixSlice<T>, sync_steps: usize) -> DMatrix<T> {
        self.reservoir_dynamics
            .record_states(&mut self.reservoir_state, input, sync_steps)
    }

    pub fn record_states_into(
        &mut self,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        result: DMatrixSliceMut<T>,
    ) {
        self.reservoir_dynamics.record_states_into(
            &mut self.reservoir_state,
            input,
            sync_steps,
            result,
        )
    }

    pub fn synchronize_and_predict<
        M: ReservoirStateMeasurement<T>,
        P: ReservoirStateProjection<T>,
    >(
        &mut self,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        predict_steps: usize,
        readout: &mut M,
        projection: &mut P,
    ) -> DMatrix<T> {
        self.reservoir_dynamics.synchronize_and_predict(
            &mut self.reservoir_state,
            input,
            sync_steps,
            predict_steps,
            readout,
            projection,
        )
    }

    pub fn synchronize_and_predict_into<
        M: ReservoirStateMeasurement<T>,
        P: ReservoirStateProjection<T>,
    >(
        &mut self,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        predict_steps: usize,
        readout: &mut M,
        projection: &mut P,
        result: DMatrixSliceMut<T>,
    ) {
        self.reservoir_dynamics.synchronize_and_predict_into(
            &mut self.reservoir_state,
            input,
            sync_steps,
            predict_steps,
            readout,
            projection,
            result,
        );
    }
}

impl<T, I, E> Clone for Reservoir<T, I, E>
where
    T: ReservoirValue + Clone,
    I: ReservoirInputProjection<T> + Clone,
    E: ReservoirTimeEvolution<T> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            reservoir_state: self.reservoir_state.clone(),
            reservoir_dynamics: self.reservoir_dynamics.clone(),
        }
    }
}
