use nalgebra::DVector;

use crate::{
    controlled_time_evolution::ControlledReservoirTimeEvolution,
    input_projection::ReservoirInputProjection, ReservoirValue,
};

use super::ControlledReservoirDynamics;

#[derive(Debug)]
pub struct ControlledReservoir<T, I, C, E>
where
    T: ReservoirValue,
    E: ControlledReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
{
    reservoir_state: DVector<T>,
    reservoir_dynamics: ControlledReservoirDynamics<T, I, C, E>,
}

impl<T, I, C, E> ControlledReservoir<T, I, C, E>
where
    T: ReservoirValue,
    E: ControlledReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
{
    pub fn reservoir_state(&self) -> &DVector<T> {
        &self.reservoir_state
    }

    pub fn reservoir_dynamics(&self) -> &ControlledReservoirDynamics<T, I, C, E> {
        &self.reservoir_dynamics
    }
}
