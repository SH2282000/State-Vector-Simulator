#cython: language_level=3, boundscheck=False, wraparound=False
cimport numpy as np
from cython.parallel import prange

cpdef apply_gate(
                np.ndarray[np.complex_t, ndim=2] psi,
                np.ndarray[np.complex_t, ndim=2] gate_arr,
                np.ndarray[np.int64_t, ndim=1] qbit_indices_steps,
                np.int64_t target,
                np.int64_t numQubits,):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t step_ind
    cdef np.complex_t alpha
    cdef np.complex_t beta
    cdef np.int64_t ij_idx
    cdef np.int64_t ij_step_idx
    cdef np.int64_t max_runtime
    cdef np.complex_t res1, res2


    max_runtime = int(2**numQubits)
    step_ind =  qbit_indices_steps[numQubits - 1 - target]
    for i in prange(0,max_runtime,2*step_ind, nogil=True):
        for j in prange(0,step_ind,1):

            ij_idx = i+j
            ij_step_idx = ij_idx + step_ind

            alpha = psi[ij_idx,0]
            beta = psi[ij_step_idx,0]
            res1 = alpha * gate_arr[0,0] + beta * gate_arr[0,1]
            res2 = alpha * gate_arr[1,0] + beta * gate_arr[1,1]
            psi[ij_idx,0] = res1
            psi[ij_step_idx,0] = res2


cpdef apply_gate_cx(np.ndarray[np.complex_t, ndim=2] psi,
                    np.ndarray[np.complex_t, ndim=2] gate_arr,
                    np.ndarray[np.int64_t, ndim=1] qbit_indices_steps,
                    np.int64_t numQubits,
                    np.int64_t control,
                    np.int64_t target
                    ):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t control_ind
    cdef np.int64_t target_step_ind
    cdef np.complex_t alpha
    cdef np.complex_t beta
    cdef np.int64_t ij_idx
    cdef np.int64_t ij_step_idx
    cdef np.int64_t max_runtime

    max_runtime = int(2**numQubits)

    control_ind = qbit_indices_steps[numQubits - 1 - control]
    target_step_ind = qbit_indices_steps[numQubits - 1 - target]
    # offset i to only look at '1' entries
    for i in prange(control_ind, max_runtime, 2*control_ind, nogil=True):
        for j in prange(0, control_ind, 1):
            if 0 == ((i + j) // target_step_ind) % 2:
                ij_idx = i + j
                ij_step_idx = ij_idx + target_step_ind

                alpha = psi[ij_idx,0]
                beta = psi[ij_step_idx,0]

                psi[ij_idx, 0] = alpha * gate_arr[0,0] + beta * gate_arr[0,1]
                psi[ij_step_idx, 0] = alpha * gate_arr[1,0] + beta * gate_arr[1,1]


cpdef apply_gate_ccx(
                    np.ndarray[np.complex_t, ndim=2] psi,
                    np.ndarray[np.complex_t, ndim=2] gate_arr,
                    np.ndarray[np.int64_t, ndim=1] qbit_indices_steps,
                    np.int64_t numQubits,
                    np.int64_t control0,
                    np.int64_t control1,
                    np.int64_t target
):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t control_step_ind_0
    cdef np.int64_t control_step_ind_1
    cdef np.int64_t target_step_ind
    cdef np.complex_t alpha
    cdef np.complex_t beta
    cdef np.int64_t ij_idx
    cdef np.int64_t ij_step_idx
    cdef np.int64_t max_runtime

    max_runtime = int(2**numQubits)


    control_step_ind_0 = qbit_indices_steps[
        numQubits - 1 - control0
        ]
    control_step_ind_1 = qbit_indices_steps[
        numQubits - 1 - control1
        ]
    target_step_ind = qbit_indices_steps[numQubits - 1 - target]
    # offset i to only look at '1' entries
    for i in prange(control_step_ind_0, max_runtime,2 * control_step_ind_0,nogil=True ):
        for j in prange(0, control_step_ind_0, 1):
            if (
                    1 == ((i + j) // control_step_ind_1) % 2
                    and 0 == ((i + j) // target_step_ind) % 2
            ):
                ij_idx = i + j
                ij_step_idx = ij_idx + target_step_ind

                alpha = psi[ij_idx,0]
                beta = psi[ij_step_idx,0]
                psi[ij_idx,0] = alpha * gate_arr[0,0] + beta * gate_arr[0,1]
                psi[ij_step_idx,0] = alpha * gate_arr[1,0] + beta * gate_arr[1,1]

                j = j + 1