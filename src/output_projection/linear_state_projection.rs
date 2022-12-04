use std::fmt::Debug;

use crate::ReservoirValue;
use nalgebra::{
    ClosedAdd, ClosedMul, ComplexField, DMatrix, DMatrixSlice, DMatrixSliceMut, DVector,
    DVectorSliceMut,
};

use super::ReservoirStateProjection;

#[derive(Clone, Debug)]
pub struct LinearStateProjection<T: ReservoirValue + ComplexField + ClosedAdd + ClosedMul> {
    w_out: DMatrix<T>,
    result: DVector<T>,
}

impl<T: ReservoirValue + ComplexField + ClosedAdd + ClosedMul> LinearStateProjection<T> {
    pub fn via_ridge_regression_nalgebra(
        beta: T,
        measured_states: &DMatrix<T>,
        target_states: DMatrixSlice<T>,
    ) -> Self {
        let dimension_measured_state = measured_states.nrows();

        let measured_states_transpose = measured_states.transpose();
        let target_state_transpose = target_states.transpose();

        let rhs = measured_states * target_state_transpose;
        let lhs = if beta == T::zero() {
            measured_states * &measured_states_transpose
        } else {
            let mut reg_matrix = DMatrix::zeros(dimension_measured_state, dimension_measured_state);
            reg_matrix.fill_diagonal(beta);
            measured_states * &measured_states_transpose + reg_matrix
        };
        let lu = nalgebra::LU::new(lhs);
        let w_out = lu.solve(&rhs).unwrap().transpose();

        Self {
            w_out,
            result: DVector::zeros(target_states.nrows()),
        }
    }

    pub fn via_tikhonov_regularization_nalgebra(
        tikhonov: &DMatrix<T>,
        measured_states: &DMatrix<T>,
        target_states: DMatrixSlice<T>,
    ) -> Self {
        let measured_states_transpose = measured_states.transpose();
        let target_state_transpose = target_states.transpose();

        let rhs = measured_states * target_state_transpose;
        let lhs = {
            let reg_matrix = &tikhonov.transpose() * tikhonov;
            measured_states * &measured_states_transpose + reg_matrix
        };
        let lu = nalgebra::LU::new(lhs);
        let w_out = lu.solve(&rhs).unwrap().transpose();

        Self {
            w_out,
            result: DVector::zeros(target_states.nrows()),
        }
    }

    fn impl_project(w_out: &DMatrix<T>, state: &DVector<T>, mut result: DVectorSliceMut<T>) {
        w_out.mul_to(state, &mut result);
    }

    fn impl_project_many(
        w_out: &DMatrix<T>,
        states: DMatrixSlice<T>,
        mut targets: DMatrixSliceMut<T>,
    ) {
        assert_eq!(w_out.nrows(), targets.nrows());
        assert_eq!(states.ncols(), targets.ncols());
        w_out.mul_to(&states, &mut targets);
    }
}

#[cfg(feature = "lapack")]
impl<T: ReservoirValue + ComplexField + ClosedAdd + ClosedMul> LinearStateProjection<T> {
    pub fn via_ridge_regression_lapack(
        beta: T,
        measured_states: &DMatrix<T>,
        target_states: DMatrixSlice<T>,
    ) -> Self {
        let dimension_measured_state = measured_states.nrows();
        let target_state_transpose = target_states.transpose();

        let rhs = measured_states * target_state_transpose;
        let mut lhs = DMatrix::from_diagonal_element(
            dimension_measured_state,
            dimension_measured_state,
            beta,
        );
        unsafe {
            blas_sys::dsyrk_(
                "U".as_ptr() as *const i8,
                "N".as_ptr() as *const i8,
                &(dimension_measured_state as i32) as *const i32,
                &(measured_states.ncols() as i32) as *const i32,
                &1.0 as *const f64,
                std::mem::transmute(measured_states.data.as_vec().as_ptr()),
                &(dimension_measured_state as i32) as *const i32,
                &1.0 as *const f64,
                std::mem::transmute(lhs.data.as_mut_slice().as_mut_ptr()),
                &(dimension_measured_state as i32) as *const i32,
            );
        }
        for i in 0..(lhs.nrows() - 1) {
            for j in 0..i {
                lhs[(i, j)] = lhs[(j, i)];
            }
        }
        let lu = nalgebra::LU::new(lhs);
        let w_out = lu.solve(&rhs).unwrap().transpose();

        Self {
            w_out,
            result: DVector::zeros(target_states.nrows()),
        }
    }
}

impl<T: ReservoirValue + ComplexField + ClosedAdd + ClosedMul> ReservoirStateProjection<T>
    for LinearStateProjection<T>
{
    fn output_dimension(&self) -> usize {
        self.result.nrows()
    }

    fn input_dimension(&self) -> usize {
        self.w_out.ncols()
    }

    fn project(&mut self, state: &DVector<T>) -> &DVector<T> {
        let vec_slice_mut = DVectorSliceMut::from(self.result.as_mut_slice());
        Self::impl_project(&self.w_out, state, vec_slice_mut);
        &self.result
    }

    fn project_into(&self, state: &DVector<T>, target: DVectorSliceMut<T>) {
        Self::impl_project(&self.w_out, state, target);
    }

    fn project_many(&self, states: DMatrixSlice<T>) -> DMatrix<T> {
        let mut targets = DMatrix::zeros(self.w_out.nrows(), states.ncols());
        Self::impl_project_many(&self.w_out, states, targets.columns_mut(0, targets.ncols()));
        targets
    }

    fn project_many_into(&self, states: DMatrixSlice<T>, targets: DMatrixSliceMut<T>) {
        Self::impl_project_many(&self.w_out, states, targets)
    }
}
