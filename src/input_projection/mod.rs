use std::fmt::Debug;

use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, DVector, DVectorSliceMut};

use crate::ReservoirValue;

pub mod default_input_projection;
pub mod identity_projection_with_embedding;
pub mod input_projection_with_embedding;

pub use default_input_projection::DefaultInputProjection;
pub use identity_projection_with_embedding::IdentityProjectionWithEmbedding;
pub use input_projection_with_embedding::InputProjectionWithEmbedding;

pub trait ReservoirInputProjection<T: ReservoirValue>: Debug + Send + Sync {
    fn output_dimensions(&self) -> usize;

    fn input_dimension(&self) -> usize;

    fn embeddings(&self) -> usize;

    fn required_input_columns(&self) -> usize;

    fn project(&mut self, input: DMatrixSlice<T>) -> &DVector<T>;

    fn project_into(&mut self, input: DMatrixSlice<T>, target: DVectorSliceMut<T>);

    fn project_many(&self, inputs: DMatrixSlice<T>) -> DMatrix<T>;

    fn project_many_into(&self, inputs: DMatrixSlice<T>, targets: DMatrixSliceMut<T>);
}

impl<T: ReservoirValue, I: ReservoirInputProjection<T>> ReservoirInputProjection<T> for Box<I> {
    fn output_dimensions(&self) -> usize {
        (**self).output_dimensions()
    }

    fn input_dimension(&self) -> usize {
        (**self).input_dimension()
    }

    fn embeddings(&self) -> usize {
        (**self).embeddings()
    }

    fn required_input_columns(&self) -> usize {
        (**self).required_input_columns()
    }

    fn project(&mut self, input: DMatrixSlice<T>) -> &DVector<T> {
        (**self).project(input)
    }

    fn project_into(&mut self, input: DMatrixSlice<T>, target: DVectorSliceMut<T>) {
        (**self).project_into(input, target);
    }

    fn project_many(&self, inputs: DMatrixSlice<T>) -> DMatrix<T> {
        (**self).project_many(inputs)
    }

    fn project_many_into(&self, inputs: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        (**self).project_many_into(inputs, targets);
    }
}

impl<T: ReservoirValue> ReservoirInputProjection<T> for Box<dyn ReservoirInputProjection<T>> {
    fn output_dimensions(&self) -> usize {
        (**self).output_dimensions()
    }

    fn input_dimension(&self) -> usize {
        (**self).input_dimension()
    }

    fn embeddings(&self) -> usize {
        (**self).embeddings()
    }

    fn required_input_columns(&self) -> usize {
        (**self).required_input_columns()
    }

    fn project(&mut self, input: DMatrixSlice<T>) -> &DVector<T> {
        (**self).project(input)
    }

    fn project_into(&mut self, input: DMatrixSlice<T>, target: DVectorSliceMut<T>) {
        (**self).project_into(input, target);
    }

    fn project_many(&self, inputs: DMatrixSlice<T>) -> DMatrix<T> {
        (**self).project_many(inputs)
    }

    fn project_many_into(&self, inputs: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        (**self).project_many_into(inputs, targets);
    }
}
