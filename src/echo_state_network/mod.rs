use std::fmt::Debug;

use nalgebra::{DMatrix, DVector, RealField};
use nalgebra_sparse::CsrMatrix;
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    thread_rng,
};

use crate::{activation_function::ActiviationFunction, ReservoirValue};

pub mod sparse_discrete_echo_state_network;
pub mod sparse_leaky_integrator_echo_state_network;

pub use sparse_discrete_echo_state_network::SparseDiscreteEchoStateNetwork;
pub use sparse_leaky_integrator_echo_state_network::SparseLeakyIntegratorEchoStateNetwork;

#[derive(Clone, Debug)]
pub struct EchoStateNetworkBuilder<T: ReservoirValue + From<f32> + RealField + SampleUniform> {
    spectral_radius: Option<T>,
    adjacency_matrix: CsrMatrix<T>,
}

impl<T: ReservoirValue + From<f32> + RealField + SampleUniform> EchoStateNetworkBuilder<T> {
    pub fn random(size: usize, average_degree: usize) -> Self {
        let link_probability: T = (average_degree as f32 / (size - 1) as f32).into();
        let mut rng = thread_rng();
        let zero_one = Uniform::new_inclusive(T::zero(), T::one());
        let plus_minus_one = Uniform::new_inclusive(-T::one(), T::one());

        let mut adjacency_matrix = DMatrix::zeros(size, size);
        for i in 0..adjacency_matrix.nrows() {
            for j in 0..adjacency_matrix.ncols() {
                if i != j && zero_one.sample(&mut rng) <= link_probability {
                    adjacency_matrix[(i, j)] = plus_minus_one.sample(&mut rng);
                }
            }
        }
        let adjacency_matrix = CsrMatrix::from(&adjacency_matrix);

        Self {
            adjacency_matrix,
            spectral_radius: None,
        }
    }

    pub fn spectral_radius(&mut self, radius: T) -> &mut Self {
        let mut rng = thread_rng();
        let plus_minus_one = Uniform::new_inclusive(-T::one(), T::one());

        // Power iteration:
        let mut random_vector = DVector::zeros(self.adjacency_matrix.nrows());
        for i in 0..random_vector.nrows() {
            random_vector[i] = plus_minus_one.sample(&mut rng);
        }

        for _ in 0..50 {
            random_vector.normalize_mut();
            random_vector = &self.adjacency_matrix * random_vector;
        }
        let spectral_radius = random_vector.norm();

        self.adjacency_matrix *= radius / spectral_radius;
        self.spectral_radius = Some(spectral_radius);
        self
    }

    pub fn build_sparse_discrete_network<A: ActiviationFunction<T>>(
        self,
        a: A,
    ) -> SparseDiscreteEchoStateNetwork<T, A> {
        SparseDiscreteEchoStateNetwork {
            adjacency_matrix: self.adjacency_matrix,
            activation_function: a,
        }
    }
}
