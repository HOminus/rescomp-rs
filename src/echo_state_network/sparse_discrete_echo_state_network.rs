use std::fmt::Debug;

use nalgebra::{DVector, DVectorSlice, RealField};
use nalgebra_sparse::CsrMatrix;
use rand::distributions::uniform::SampleUniform;

use crate::{
    activation_function::ActiviationFunction, time_evolution::ReservoirTimeEvolution,
    ReservoirValue,
};

#[derive(Clone)]
pub struct SparseDiscreteEchoStateNetwork<
    T: ReservoirValue + From<f32> + RealField + SampleUniform,
    A: ActiviationFunction<T>,
> {
    pub(super) adjacency_matrix: CsrMatrix<T>,
    pub(super) activation_function: A,
}

impl<T: ReservoirValue + From<f32> + RealField + SampleUniform, A: ActiviationFunction<T>> Debug
    for SparseDiscreteEchoStateNetwork<T, A>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EchoStateNetwork{{ {:?} }}", self.adjacency_matrix)
    }
}

impl<T: ReservoirValue + From<f32> + RealField + SampleUniform, A: ActiviationFunction<T>>
    ReservoirTimeEvolution<T> for SparseDiscreteEchoStateNetwork<T, A>
{
    fn input_dimension(&self) -> usize {
        self.adjacency_matrix.nrows()
    }

    fn output_dimension(&self) -> usize {
        self.adjacency_matrix.nrows()
    }

    fn time_evolution(&self, state: &mut DVector<T>, input: DVectorSlice<T>) {
        let combined_state = &self.adjacency_matrix * &(*state) + input;
        for (index, (s, e)) in state
            .as_mut_slice()
            .iter_mut()
            .zip(combined_state.as_slice().iter())
            .enumerate()
        {
            *s = self.activation_function.invoke(index, *e);
        }
    }
}
