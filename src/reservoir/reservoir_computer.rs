use std::fmt::Debug;

use crate::input_projection::ReservoirInputProjection;
use crate::output_projection::ReservoirStateProjection;
use crate::state_measurement::ReservoirStateMeasurement;
use crate::time_evolution::ReservoirTimeEvolution;
use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, DVector};

use super::Reservoir;
use crate::ReservoirValue;

#[derive(Debug)]
pub struct ReservoirComputer<T, I, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    E: ReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    pub(crate) reservoir: Reservoir<T, I, E>,
    pub(crate) reservoir_state_measurement: M,
    pub(crate) reservoir_state_projection: P,
}

impl<T, I, E, M, P> ReservoirComputer<T, I, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    E: ReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    pub fn state(&self) -> &DVector<T> {
        self.reservoir.state()
    }

    pub fn state_input_projection(&self) -> &I {
        self.reservoir.input_projection()
    }

    pub fn state_measurement(&self) -> &M {
        &self.reservoir_state_measurement
    }

    pub fn state_projection(&self) -> &P {
        &self.reservoir_state_projection
    }

    pub fn synchronize_and_predict(
        &mut self,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        predict_steps: usize,
    ) -> DMatrix<T> {
        self.reservoir.synchronize_and_predict(
            input,
            sync_steps,
            predict_steps,
            &mut self.reservoir_state_measurement,
            &mut self.reservoir_state_projection,
        )
    }

    pub fn synchronize_and_predict_into(
        &mut self,
        input: DMatrixSlice<T>,
        sync_steps: usize,
        predict_steps: usize,
        result: DMatrixSliceMut<T>,
    ) {
        self.reservoir.synchronize_and_predict_into(
            input,
            sync_steps,
            predict_steps,
            &mut self.reservoir_state_measurement,
            &mut self.reservoir_state_projection,
            result,
        );
    }
}

impl<T, I, E, M, P> Clone for ReservoirComputer<T, I, E, M, P>
where
    T: ReservoirValue + Clone,
    I: ReservoirInputProjection<T> + Clone,
    E: ReservoirTimeEvolution<T> + Clone,
    M: ReservoirStateMeasurement<T> + Clone,
    P: ReservoirStateProjection<T> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            reservoir: self.reservoir.clone(),
            reservoir_state_measurement: self.reservoir_state_measurement.clone(),
            reservoir_state_projection: self.reservoir_state_projection.clone(),
        }
    }
}
