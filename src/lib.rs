//! It is recommended to use f64s since f32s are numerically not stable enough.

use std::fmt::{Debug, Display};

use nalgebra::{RealField, Scalar};
use num_traits::Float;

pub mod activation_function;
pub mod controlled_reservoir;
pub mod controlled_time_evolution;
pub mod echo_state_network;
pub mod input_projection;
pub mod output_projection;
pub mod reservoir;
pub mod state_measurement;
pub mod time_evolution;

pub use reservoir::{Reservoir, ReservoirComputer, ReservoirComputerDynamics, ReservoirDynamics};

#[cfg(not(feature = "lapack"))]
pub trait ReservoirValue: Display + Scalar + Copy + Debug + Float + RealField {}

#[cfg(feature = "lapack")]
pub trait ReservoirValue:
    Display + Scalar + Copy + Debug + Float + RealField + nalgebra_lapack::LUScalar
{
}

impl ReservoirValue for f32 {}
impl ReservoirValue for f64 {}
