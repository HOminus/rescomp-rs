use std::fmt::Debug;

use nalgebra::{
    ClosedAdd, ClosedMul, DMatrix, DMatrixSlice, DMatrixSliceMut, DVector, DVectorSliceMut,
};

use super::ReservoirInputProjection;
use crate::ReservoirValue;

#[derive(Clone, Debug)]
pub struct IdentityProjectionWithEmbedding<T: ReservoirValue + ClosedAdd + ClosedMul> {
    input_dimensions: usize,
    embeddings: usize,
    stride: usize,
    result: DVector<T>,
}

impl<T: ReservoirValue + ClosedAdd + ClosedMul> IdentityProjectionWithEmbedding<T> {
    pub fn new(system_dim: usize, embeddings: usize, stride: usize) -> Self {
        if embeddings > 0 {
            assert_ne!(stride, 0);
        }
        Self {
            input_dimensions: system_dim,
            embeddings,
            stride,
            result: DVector::zeros(system_dim * (embeddings + 1)),
        }
    }

    fn impl_project(input: DMatrixSlice<T>, stride: usize, mut result: DVectorSliceMut<T>) {
        let system_dimension = input.nrows();
        let mut iter = input.column_iter();
        let mut counter = 0;
        while let Some(column_slice) = iter.next() {
            let mut temp_slice = result.rows_mut(counter * system_dimension, system_dimension);
            temp_slice.copy_from(&column_slice);

            if stride > 1 {
                iter.nth(stride - 2);
            }
            counter += 1;
        }
    }

    fn impl_project_many(input: DMatrixSlice<T>, stride: usize, mut result: DMatrixSliceMut<T>) {
        let input_dim = input.nrows();
        let embeddings = result.nrows() / input.nrows() - 1;
        for i in 0..result.ncols() {
            result
                .column_mut(i)
                .rows_mut(0, input_dim)
                .copy_from(&input.column(i));

            for e in 1..=embeddings {
                result
                    .column_mut(i)
                    .rows_mut(e * input_dim, input_dim)
                    .copy_from(&input.column(i + e * stride));
            }
        }
    }
}

impl<T: ReservoirValue + ClosedAdd + ClosedMul> ReservoirInputProjection<T>
    for IdentityProjectionWithEmbedding<T>
{
    fn output_dimensions(&self) -> usize {
        self.result.nrows()
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
        let target_slice = self.result.column_mut(0);
        Self::impl_project(input, self.stride, target_slice);
        &self.result
    }

    fn project_into(&mut self, input: DMatrixSlice<T>, target: DVectorSliceMut<T>) {
        assert_eq!(input.ncols(), self.required_input_columns());
        assert_eq!(target.nrows(), self.output_dimensions());
        Self::impl_project(input, self.stride, target);
    }

    fn project_many(&self, inputs: DMatrixSlice<T>) -> DMatrix<T> {
        let mut result = DMatrix::zeros(
            self.output_dimensions(),
            1 + inputs.ncols() - self.required_input_columns(),
        );
        Self::impl_project_many(inputs, self.stride, result.columns_mut(0, result.ncols()));
        result
    }

    fn project_many_into(&self, inputs: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        Self::impl_project_many(inputs, self.stride, targets);
    }
}

#[cfg(test)]
mod tests {
    use super::IdentityProjectionWithEmbedding;
    use crate::input_projection::ReservoirInputProjection;
    use nalgebra::DMatrix;

    #[test]
    fn identity_input_no_embedding() {
        let mut identity_projection = IdentityProjectionWithEmbedding::<f64>::new(3, 0, 0);

        let data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let input_matrix = DMatrix::from_vec(3, 3, data.clone());

        for i in [0, 1, 2] {
            let project_result = identity_projection.project(input_matrix.columns(i, 1));
            assert_eq!(project_result.as_slice(), &data[(i * 3)..(i + 1) * 3]);
        }

        let project_many = identity_projection.project_many(input_matrix.columns(0, 3));
        assert_eq!(project_many.column(0).as_slice(), &[1., 2., 3.]);
        assert_eq!(project_many.column(1).as_slice(), &[4., 5., 6.]);
        assert_eq!(project_many.column(2).as_slice(), &[7., 8., 9.]);
    }

    #[test]
    fn identity_input_one_embedding() {
        let mut identity_projection = IdentityProjectionWithEmbedding::<f64>::new(3, 1, 1);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let input_matrix = DMatrix::from_vec(3, 6, data.clone());

        for i in [0, 1, 2, 3, 4] {
            let project_result = identity_projection.project(input_matrix.columns(i, 2));
            assert_eq!(project_result.as_slice(), &data[(i * 3)..(i + 2) * 3]);
        }

        let project_many = identity_projection.project_many(input_matrix.columns(0, 6));
        assert_eq!(project_many.column(0).as_slice(), &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(project_many.column(1).as_slice(), &[4., 5., 6., 7., 8., 9.]);
        assert_eq!(
            project_many.column(2).as_slice(),
            &[7., 8., 9., 10., 11., 12.]
        );
        assert_eq!(
            project_many.column(3).as_slice(),
            &[10., 11., 12., 13., 14., 15.]
        );
        assert_eq!(
            project_many.column(4).as_slice(),
            &[13., 14., 15., 16., 17., 18.]
        );
    }

    #[test]
    fn identity_input_many_embeddings() {
        let mut identity_projection = IdentityProjectionWithEmbedding::<f64>::new(3, 4, 1);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let input_matrix = DMatrix::from_vec(3, 6, data.clone());

        for i in [0, 1] {
            let project_result = identity_projection.project(input_matrix.columns(i, 5));
            assert_eq!(project_result.as_slice(), &data[(i * 3)..(i + 5) * 3]);
        }

        let project_many = identity_projection.project_many(input_matrix.columns(0, 6));
        assert_eq!(
            project_many.column(0).as_slice(),
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
        );
        assert_eq!(
            project_many.column(1).as_slice(),
            &[4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.]
        );
    }

    #[test]
    fn identity_input_many_embeddings_with_stride() {
        let mut identity_projection = IdentityProjectionWithEmbedding::<f64>::new(3, 2, 3);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 20.,
            21., 22.,
        ];
        let input_matrix = DMatrix::from_vec(3, 7, data);

        let project_result = identity_projection
            .project(input_matrix.columns(0, identity_projection.required_input_columns()));
        assert_eq!(
            project_result.as_slice(),
            &[1., 2., 3., 10., 11., 12., 20., 21., 22.]
        );

        let project_many = identity_projection.project_many(input_matrix.columns(0, 7));
        assert_eq!(
            project_many.column(0).as_slice(),
            &[1., 2., 3., 10., 11., 12., 20., 21., 22.]
        );
    }

    #[test]
    #[should_panic]
    fn identity_input_wrong_input_dimensions_no_stride() {
        let mut identity_projection = IdentityProjectionWithEmbedding::<f64>::new(3, 2, 0);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 20.,
            21., 22.,
        ];
        let input_matrix = DMatrix::from_vec(3, 7, data);

        identity_projection.project(input_matrix.columns(0, 5));
    }

    #[test]
    #[should_panic]
    fn identity_input_wrong_input_dimensions_with_stride() {
        let mut identity_projection = IdentityProjectionWithEmbedding::<f64>::new(3, 2, 2);

        let data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 20.,
            21., 22.,
        ];
        let input_matrix = DMatrix::from_vec(3, 7, data);

        identity_projection.project(input_matrix.columns(0, 2));
    }
}
