use std::fmt::Debug;

use crate::input_projection::ReservoirInputProjection;
use crate::output_projection::ReservoirStateProjection;
use crate::state_measurement::ReservoirStateMeasurement;
use crate::time_evolution::ReservoirTimeEvolution;
use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, DVector};

use super::ReservoirDynamics;
use crate::ReservoirValue;

#[derive(Debug)]
pub struct ReservoirComputerDynamics<T, I, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    E: ReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    reservoir_dynamics: ReservoirDynamics<T, I, E>,
    reservoir_state_measurement: M,
    reservoir_state_projection: P,
}

impl<T, I, E, M, P> ReservoirComputerDynamics<T, I, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    E: ReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    pub fn into_parts(self) -> (I, E, M, P) {
        let (i, e) = self.reservoir_dynamics.into_parts();
        (
            i,
            e,
            self.reservoir_state_measurement,
            self.reservoir_state_projection,
        )
    }

    pub fn split_reservoir_dynamics(self) -> (ReservoirDynamics<T, I, E>, M, P) {
        (
            self.reservoir_dynamics,
            self.reservoir_state_measurement,
            self.reservoir_state_projection,
        )
    }

    pub fn synchronize_and_predict(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        preict_steps: usize,
    ) -> DMatrix<T> {
        self.reservoir_dynamics.synchronize_and_predict(
            state,
            input,
            sync_steps,
            preict_steps,
            &mut self.reservoir_state_measurement,
            &mut self.reservoir_state_projection,
        )
    }

    pub fn synchronize_and_predict_into(
        &mut self,
        state: &mut DVector<T>,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        predict_steps: usize,
        result: DMatrixSliceMut<T>,
    ) {
        self.reservoir_dynamics.synchronize_and_predict_into(
            state,
            input,
            sync_steps,
            predict_steps,
            &mut self.reservoir_state_measurement,
            &mut self.reservoir_state_projection,
            result,
        );
    }
}

impl<T, I, E, M, P> Clone for ReservoirComputerDynamics<T, I, E, M, P>
where
    T: ReservoirValue + Clone,
    I: ReservoirInputProjection<T> + Clone,
    E: ReservoirTimeEvolution<T> + Clone,
    M: ReservoirStateMeasurement<T> + Clone,
    P: ReservoirStateProjection<T> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            reservoir_dynamics: self.reservoir_dynamics.clone(),
            reservoir_state_measurement: self.reservoir_state_measurement.clone(),
            reservoir_state_projection: self.reservoir_state_projection.clone(),
        }
    }
}
