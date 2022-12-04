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
pub struct InputProjectionWithEmbedding<T: ReservoirValue + SampleUniform + ClosedAdd + ClosedMul> {
    w_in: DMatrix<T>,
    input_dimensions: usize,
    embeddings: usize,
    stride: usize,
    temporary: DVector<T>,
    result: DVector<T>,
}

impl<T: ReservoirValue + SampleUniform + ClosedAdd + ClosedMul> InputProjectionWithEmbedding<T> {
    pub fn new_random(
        system_dim: usize,
        output_dim: usize,
        embeddings: usize,
        stride: usize,
    ) -> Self {
        assert_ne!(stride, 0);

        let input_dim = system_dim * (1 + embeddings);
        let mut w_in = DMatrix::zeros(output_dim, input_dim);

        let mut rnd = thread_rng();
        let choice_distribution = Uniform::new(0, input_dim);
        let value_distribution = Uniform::new(-T::one(), T::one());

        for mut row in w_in.row_iter_mut() {
            let choice = choice_distribution.sample(&mut rnd);
            let value = value_distribution.sample(&mut rnd);
            row[choice] = value;
        }

        Self {
            w_in,
            input_dimensions: system_dim,
            embeddings,
            stride,
            temporary: DVector::zeros(input_dim),
            result: DVector::zeros(output_dim),
        }
    }

    pub fn new_with_matrix(matrix: DMatrix<T>, embeddings: usize, stride: usize) -> Self {
        let input_dimensions = matrix.ncols() / (1 + embeddings);
        let output_dimensions = matrix.nrows();

        Self {
            temporary: DVector::zeros(matrix.ncols()),
            w_in: matrix,
            input_dimensions,
            embeddings,
            stride,
            result: DVector::zeros(output_dimensions),
        }
    }

    fn impl_project(
        w_in: &DMatrix<T>,
        input: DMatrixSlice<T>,
        embeddings: usize,
        stride: usize,
        mut temporary: DVectorSliceMut<T>,
        mut result: DVectorSliceMut<T>,
    ) {
        let input_dim = input.nrows();
        temporary.rows_mut(0, input_dim).copy_from(&input.column(0));

        for e in 1..=embeddings {
            temporary
                .rows_mut(e * input_dim, input_dim)
                .copy_from(&input.column(e * stride));
        }
        w_in.mul_to(&temporary, &mut result);
    }

    fn impl_project_many(
        w_in: &DMatrix<T>,
        input: DMatrixSlice<T>,
        embeddings: usize,
        stride: usize,
        mut temporary: DVectorSliceMut<T>,
        mut result: DMatrixSliceMut<T>,
    ) {
        let input_dim = input.nrows();
        for i in 0..result.ncols() {
            temporary.rows_mut(0, input_dim).copy_from(&input.column(i));

            for e in 1..=embeddings {
                temporary
                    .rows_mut(e * input_dim, input_dim)
                    .copy_from(&input.column(i + e * stride));
            }
            w_in.mul_to(&temporary, &mut result.column_mut(i));
        }
    }
}

impl<T: ReservoirValue + SampleUniform + ClosedAdd + ClosedMul> ReservoirInputProjection<T>
    for InputProjectionWithEmbedding<T>
{
    fn output_dimensions(&self) -> usize {
        self.w_in.nrows()
    }

    fn input_dimension(&self) -> usize {
        self.input_dimensions
    }

    fn embeddings(&self) -> usize {
        self.embeddings
    }

    fn required_input_columns(&self) -> usize {
        1 + self.embeddings * self.stride
    }

    fn project(&mut self, input: DMatrixSlice<T>) -> &DVector<T> {
        assert_eq!(input.ncols(), self.required_input_columns());
        let temp_slice = self.temporary.column_mut(0);
        let target_slice = self.result.column_mut(0);
        Self::impl_project(
            &self.w_in,
            input,
            self.embeddings,
            self.stride,
            temp_slice,
            target_slice,
        );
        &self.result
    }

    fn project_into(&mut self, input: DMatrixSlice<T>, target: DVectorSliceMut<T>) {
        assert_eq!(input.ncols(), self.required_input_columns());
        assert_eq!(target.nrows(), self.output_dimensions());

        let temp_slice = self.temporary.column_mut(0);
        Self::impl_project(
            &self.w_in,
            input,
            self.embeddings,
            self.stride,
            temp_slice,
            target,
        );
    }

    fn project_many(&self, inputs: DMatrixSlice<T>) -> DMatrix<T> {
        let mut result = DMatrix::zeros(
            self.output_dimensions(),
            1 + inputs.ncols() - self.required_input_columns(),
        );
        let mut new_temp = DVector::zeros(self.temporary.nrows());
        Self::impl_project_many(
            &self.w_in,
            inputs,
            self.embeddings,
            self.stride,
            new_temp.column_mut(0),
            result.columns_mut(0, result.ncols()),
        );
        result
    }

    fn project_many_into(&self, inputs: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        let mut new_temp = DVector::zeros(self.temporary.nrows());
        Self::impl_project_many(
            &self.w_in,
            inputs,
            self.embeddings,
            self.stride,
            new_temp.column_mut(0),
            targets,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::InputProjectionWithEmbedding;
    use crate::input_projection::ReservoirInputProjection;
    use nalgebra::DMatrix;

    #[test]
    fn input_projection_with_no_embedding() {
        let input_projection_matrix = DMatrix::from_vec(2, 3, vec![1., 2., 3., -4., -5., 6.]);
        //1 3 -5
        //2 -4 6
        let mut input_projection =
            InputProjectionWithEmbedding::new_with_matrix(input_projection_matrix, 0, 0);
        assert_eq!(input_projection.input_dimension(), 3);
        assert_eq!(input_projection.embeddings(), 0);
        assert_eq!(input_projection.required_input_columns(), 1);

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let input_matrix = DMatrix::from_vec(3, 3, data);

        let project_result = input_projection.project(input_matrix.columns(0, 1));
        assert_eq!(project_result.as_slice(), &[-8., 12.]);
        let project_result = input_projection.project(input_matrix.columns(1, 1));
        assert_eq!(project_result.as_slice(), &[-11., 24.]);
        let project_result = input_projection.project(input_matrix.columns(2, 1));
        assert_eq!(project_result.as_slice(), &[-14., 36.]);

        let project_many = input_projection.project_many(input_matrix.columns(0, 3));
        assert_eq!(project_many.column(0).as_slice(), &[-8., 12.]);
        assert_eq!(project_many.column(1).as_slice(), &[-11., 24.]);
        assert_eq!(project_many.column(2).as_slice(), &[-14., 36.]);
    }

    #[test]
    fn input_projection_with_one_embedding() {
        let mat_data = vec![1., 2., 3., -4., -5., 6., 0., -1., 1., 2., -3., 2.];
        let input_projection_matrix = DMatrix::from_vec(2, 6, mat_data);
        //1  3 -5  0 1 -3
        //2 -4 -6 -1 2  2
        let mut input_projection =
            InputProjectionWithEmbedding::new_with_matrix(input_projection_matrix.clone(), 1, 1);
        assert_eq!(input_projection.input_dimension(), 3);
        assert_eq!(input_projection.embeddings(), 1);
        assert_eq!(input_projection.required_input_columns(), 2);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        // 1 4 7 10 13 16
        // 2 5 8 11 14 17
        // 3 6 9 12 15 18
        let input_matrix = DMatrix::from_vec(3, 6, data);

        let data = vec![
            1., 2., 3., 4., 5., 6., 4., 5., 6., 7., 8., 9., 7., 8., 9., 10., 11., 12., 10., 11.,
            12., 13., 14., 15., 13., 14., 15., 16., 17., 18.,
        ];
        let result = input_projection_matrix * &DMatrix::from_vec(6, 5, data);

        let project_result = input_projection.project(input_matrix.columns(0, 2));
        assert_eq!(project_result.as_slice(), result.column(0).as_slice());
        let project_result = input_projection.project(input_matrix.columns(1, 2));
        assert_eq!(project_result.as_slice(), result.column(1).as_slice());
        let project_result = input_projection.project(input_matrix.columns(2, 2));
        assert_eq!(project_result.as_slice(), result.column(2).as_slice());
        let project_result = input_projection.project(input_matrix.columns(3, 2));
        assert_eq!(project_result.as_slice(), result.column(3).as_slice());
        let project_result = input_projection.project(input_matrix.columns(4, 2));
        assert_eq!(project_result.as_slice(), result.column(4).as_slice());

        let project_many = input_projection.project_many(input_matrix.columns(0, 6));
        assert_eq!(
            project_many.column(0).as_slice(),
            result.column(0).as_slice()
        );
        assert_eq!(
            project_many.column(1).as_slice(),
            result.column(1).as_slice()
        );
        assert_eq!(
            project_many.column(2).as_slice(),
            result.column(2).as_slice()
        );
        assert_eq!(
            project_many.column(3).as_slice(),
            result.column(3).as_slice()
        );
        assert_eq!(
            project_many.column(4).as_slice(),
            result.column(4).as_slice()
        );
    }

    #[test]
    fn input_projection_with_many_embeddings() {
        let mat_data = vec![
            1., 2., 3., -4., -5., 6., 0., -1., 1., 2., -3., 2., 0.5, -1.5, 2., 0., -4., 2.,
        ];
        let input_projection_matrix = DMatrix::from_vec(2, 9, mat_data);
        //1  3 -5  0 1 -3 .5   2 -4
        //2 -4 -6 -1 2  2 -1.5 0  2
        let mut input_projection =
            InputProjectionWithEmbedding::new_with_matrix(input_projection_matrix.clone(), 2, 1);
        assert_eq!(input_projection.input_dimension(), 3);
        assert_eq!(input_projection.embeddings(), 2);
        assert_eq!(input_projection.required_input_columns(), 3);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        // 1 4 7 10 13 16
        // 2 5 8 11 14 17
        // 3 6 9 12 15 18
        let input_matrix = DMatrix::from_vec(3, 6, data);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 4., 5., 6., 7., 8., 9., 10., 11., 12., 7., 8., 9.,
            10., 11., 12., 13., 14., 15., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let result = input_projection_matrix * &DMatrix::from_vec(9, 4, data);

        let project_result = input_projection.project(input_matrix.columns(0, 3));
        assert_eq!(project_result.as_slice(), result.column(0).as_slice());
        let project_result = input_projection.project(input_matrix.columns(1, 3));
        assert_eq!(project_result.as_slice(), result.column(1).as_slice());
        let project_result = input_projection.project(input_matrix.columns(2, 3));
        assert_eq!(project_result.as_slice(), result.column(2).as_slice());
        let project_result = input_projection.project(input_matrix.columns(3, 3));
        assert_eq!(project_result.as_slice(), result.column(3).as_slice());

        let project_many = input_projection.project_many(input_matrix.columns(0, 6));
        assert_eq!(
            project_many.column(0).as_slice(),
            result.column(0).as_slice()
        );
        assert_eq!(
            project_many.column(1).as_slice(),
            result.column(1).as_slice()
        );
        assert_eq!(
            project_many.column(2).as_slice(),
            result.column(2).as_slice()
        );
        assert_eq!(
            project_many.column(3).as_slice(),
            result.column(3).as_slice()
        );
    }

    #[test]
    fn input_projection_with_many_embeddings_with_stride() {
        let mat_data = vec![
            1., 2., 3., -4., -5., 6., 0., -1., 1., 2., -3., 2., 0.5, -1.5, 2., 0., -4., 2.,
        ];
        let input_projection_matrix = DMatrix::from_vec(2, 9, mat_data);
        //1  3 -5  0 1 -3 .5   2 -4
        //2 -4 -6 -1 2  2 -1.5 0  2
        let mut input_projection =
            InputProjectionWithEmbedding::new_with_matrix(input_projection_matrix.clone(), 2, 2);
        assert_eq!(input_projection.input_dimension(), 3);
        assert_eq!(input_projection.embeddings(), 2);
        assert_eq!(input_projection.required_input_columns(), 5);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 20.,
            21., 22.,
        ];
        // 1 4 7 10 13 16 19
        // 2 5 8 11 14 17 20
        // 3 6 9 12 15 18 21
        let input_matrix = DMatrix::from_vec(3, 7, data);

        let data = vec![
            1., 2., 3., 7., 8., 9., 13., 14., 15., 4., 5., 6., 10., 11., 12., 16., 17., 18., 7.,
            8., 9., 13., 14., 15., 20., 21., 22.,
        ];
        let result = input_projection_matrix * &DMatrix::from_vec(9, 3, data);

        let project_result = input_projection.project(input_matrix.columns(0, 5));
        assert_eq!(project_result.as_slice(), result.column(0).as_slice());

        let project_result = input_projection.project(input_matrix.columns(1, 5));
        assert_eq!(project_result.as_slice(), result.column(1).as_slice());

        let project_result = input_projection.project(input_matrix.columns(2, 5));
        assert_eq!(project_result.as_slice(), result.column(2).as_slice());

        let project_many = input_projection.project_many(input_matrix.columns(0, 7));
        assert_eq!(project_many.ncols(), 3);
        assert_eq!(
            project_many.column(0).as_slice(),
            result.column(0).as_slice()
        );
        assert_eq!(
            project_many.column(1).as_slice(),
            result.column(1).as_slice()
        );
        assert_eq!(
            project_many.column(2).as_slice(),
            result.column(2).as_slice()
        );
    }
}
