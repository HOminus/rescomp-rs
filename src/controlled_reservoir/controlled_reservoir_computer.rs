use crate::{
    controlled_time_evolution::ControlledReservoirTimeEvolution,
    input_projection::ReservoirInputProjection, output_projection::ReservoirStateProjection,
    state_measurement::ReservoirStateMeasurement, ReservoirValue,
};

use super::ControlledReservoir;

#[derive(Debug)]
pub struct ControlledReservoirComputer<T, I, C, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
    E: ControlledReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    reservoir: ControlledReservoir<T, I, C, E>,
    reservoir_state_measurement: M,
    reservoir_state_projection: P,
}

impl<T, I, C, E, M, P> ControlledReservoirComputer<T, I, C, E, M, P>
where
    T: ReservoirValue,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
    E: ControlledReservoirTimeEvolution<T>,
    M: ReservoirStateMeasurement<T>,
    P: ReservoirStateProjection<T>,
{
    pub fn controlled_reservoir(&self) -> &ControlledReservoir<T, I, C, E> {
        &self.reservoir
    }

    pub fn state_measurement(&self) -> &M {
        &self.reservoir_state_measurement
    }

    pub fn state_projection(&self) -> &P {
        &self.reservoir_state_projection
    }
}
