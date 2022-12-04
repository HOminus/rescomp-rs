use std::fmt::Debug;

use nalgebra::{
    ClosedAdd, ClosedMul, DMatrix, DMatrixSlice, DMatrixSliceMut, DVector, DVectorSliceMut,
};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    thread_rng,
};

use super::ReservoirInputProjection;
use crate::ReservoirValue;

#[derive(Clone, Debug)]
pub struct DefaultInputProjection<T: ReservoirValue + SampleUniform + ClosedAdd + ClosedMul> {
    w_in: DMatrix<T>,
    result: DVector<T>,
}

impl<T: ReservoirValue + SampleUniform + ClosedAdd + ClosedMul> DefaultInputProjection<T> {
    pub fn new_random(input_dim: usize, output_dim: usize, input_strength: T) -> Self {
        let mut w_in = DMatrix::zeros(output_dim, input_dim);

        let mut rnd = thread_rng();
        let choice_distribution = Uniform::new(0, input_dim);
        let value_distribution = Uniform::new(-T::one(), T::one());

        for mut row in w_in.row_iter_mut() {
            let choice = choice_distribution.sample(&mut rnd);
            let value = value_distribution.sample(&mut rnd);
            row[choice] = input_strength * value;
        }

        Self {
            w_in,
            result: DVector::zeros(output_dim),
        }
    }

    pub fn new_with_matrix(matrix: DMatrix<T>) -> Self {
        let output_dim = matrix.nrows();
        Self {
            w_in: matrix,
            result: DVector::zeros(output_dim),
        }
    }

    fn impl_project(w_in: &DMatrix<T>, input: DMatrixSlice<T>, mut result: DVectorSliceMut<T>) {
        assert_eq!(input.ncols(), 1);
        w_in.mul_to(&input, &mut result)
    }

    fn impl_project_many(
        w_in: &DMatrix<T>,
        input: DMatrixSlice<T>,
        mut result: DMatrixSliceMut<T>,
    ) {
        assert_eq!(input.ncols(), result.ncols());
        w_in.mul_to(&input, &mut result);
    }
}

impl<T: ReservoirValue + SampleUniform + ClosedAdd + ClosedMul> ReservoirInputProjection<T>
    for DefaultInputProjection<T>
{
    fn output_dimensions(&self) -> usize {
        self.w_in.nrows()
    }

    fn input_dimension(&self) -> usize {
        self.w_in.ncols()
    }

    fn embeddings(&self) -> usize {
        1
    }

    fn required_input_columns(&self) -> usize {
        1
    }

    fn project(&mut self, input: DMatrixSlice<T>) -> &DVector<T> {
        let output_dimension = self.output_dimensions();
        let target = DVectorSliceMut::from_slice(self.result.as_mut_slice(), output_dimension);
        Self::impl_project(&self.w_in, input, target);
        &self.result
    }

    fn project_into(&mut self, input: DMatrixSlice<T>, target: DVectorSliceMut<T>) {
        assert_eq!(self.required_input_columns(), input.ncols());
        assert_eq!(self.output_dimensions(), target.nrows());
        Self::impl_project(&self.w_in, input, target);
    }

    fn project_many(&self, inputs: DMatrixSlice<T>) -> DMatrix<T> {
        let mut result = DMatrix::zeros(self.w_in.nrows(), inputs.ncols());
        Self::impl_project_many(&self.w_in, inputs, result.columns_mut(0, result.ncols()));
        result
    }

    fn project_many_into(&self, inputs: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        Self::impl_project_many(&self.w_in, inputs, targets);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use super::DefaultInputProjection;
    use crate::input_projection::ReservoirInputProjection;

    #[test]
    fn default_input_projection_basic_test() {
        let input_projection_matrix = DMatrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        //1 3 5
        //2 4 6
        let mut input_projection = DefaultInputProjection::new_with_matrix(input_projection_matrix);

        let data = DMatrix::from_vec(3, 1, vec![0., 1., 2.]);

        let projection_result = input_projection.project(data.columns(0, 1));
        assert_eq!(projection_result.as_slice(), &[13., 16.]);

        let data = vec![0., 1., 2., -1., 0., -2., 3., 0., 0.];
        let data = DMatrix::from_vec(3, 3, data);
        let projection_results = input_projection.project_many(data.columns(0, 3));
        assert_eq!(projection_results.column(0).as_slice(), &[13., 16.]);
        assert_eq!(projection_results.column(1).as_slice(), &[-11., -14.]);
        assert_eq!(projection_results.column(2).as_slice(), &[3., 6.]);
    }
}
