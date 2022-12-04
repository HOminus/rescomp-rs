use crate::{
    controlled_time_evolution::ControlledReservoirTimeEvolution,
    input_projection::ReservoirInputProjection, output_projection::ReservoirStateProjection,
    state_measurement::ReservoirStateMeasurement, ReservoirValue,
};

use super::ControlledReservoirDynamics;

#[derive(Debug)]
pub struct ControlledReservoirComputerDynamics<T, I, C, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
    E: ControlledReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    reservoir_dynamics: ControlledReservoirDynamics<T, I, C, E>,
    reservoir_state_measurement: M,
    reservoir_state_projection: P,
}

impl<T, I, C, E, M, P> ControlledReservoirComputerDynamics<T, I, C, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
    E: ControlledReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    pub fn reservoir_dynamics(&self) -> &ControlledReservoirDynamics<T, I, C, E> {
        &self.reservoir_dynamics
    }

    pub fn state_measurement(&self) -> &M {
        &self.reservoir_state_measurement
    }

    pub fn state_projection(&self) -> &P {
        &self.reservoir_state_projection
    }
}
