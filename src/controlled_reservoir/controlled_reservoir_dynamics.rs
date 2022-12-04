use std::marker::PhantomData;

use crate::{
    controlled_time_evolution::ControlledReservoirTimeEvolution,
    input_projection::ReservoirInputProjection, ReservoirValue,
};

#[derive(Debug)]
pub struct ControlledReservoirDynamics<T, I, C, E>
where
    T: ReservoirValue,
    E: ControlledReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
{
    reservoir_input_projection: I,
    reservoir_controlled_input_projection: C,
    reservoir_time_evolution: E,
    _phantom: PhantomData<T>,
}

impl<T, I, C, E> ControlledReservoirDynamics<T, I, C, E>
where
    T: ReservoirValue,
    E: ControlledReservoirTimeEvolution<T>,
    I: ReservoirInputProjection<T>,
    C: ReservoirInputProjection<T>,
{
    pub fn input_projection(&self) -> &I {
        &self.reservoir_input_projection
    }

    pub fn controlled_input_projection(&self) -> &C {
        &self.reservoir_controlled_input_projection
    }

    pub fn time_evolution(&self) -> &E {
        &self.reservoir_time_evolution
    }
}
