import numpy as np
from scipy.sparse import csr_array, identity, kron

from QCP.template.apply_gate import apply_gate, apply_gate_cx, apply_gate_ccx
from QCP.parser.parseQCP import QCPcircuit
from QCP.template.constants import (
    X_GATE,
    H_GATE,
    Y_GATE,
    Z_GATE,
    T_GATE,
    TDG_GATE,
    S_GATE,
    SDG_GATE,
    SQRT_ISWAP,
)


class SVtemplate:
    circ = None

    def __init__(
        self,
        circ: QCPcircuit,
        noise: dict = {
            "bitflip": 0,
            "phaseflip": 0,
            "amplitude_damping": (1, 0),
        },
    ) -> None:
        self.circ = circ
        self.psi = np.zeros((2**self.circ.numQubits, 1), dtype=np.complex128)
        self.psi[0] = 1
        self.qbit_indices_steps = np.array(
            [
                int((2**self.circ.numQubits) / (2 ** (i + 1)))
                for i in range(self.circ.numQubits)
            ]
        )
        self.noise = noise
        self.has_error = self.noise["bitflip"] or self.noise["phaseflip"]

    def iterate_circ(self):
        if self.circ is None:
            raise Exception("circ is None")
        for gates in self.circ.gates:
            # can parallelize this
            for gate in gates:
                # applies changes to the statevector inside the function
                getattr(self, gate.name)(gate)

            # if self.has_error and False:
            #    self.apply_error_naive()

    def set_psi(self, psi):
        self.psi = np.array(psi, dtype=np.complex128, ndmin=2).T

    def simulate(self):
        self.has_error = self.noise["bitflip"] or self.noise["phaseflip"]
        self.iterate_circ()

        return self.psi

    def reset(self):
        self.psi = np.zeros((2**self.circ.numQubits, 1), dtype=complex)
        self.psi[0] = 1

    def get_probabilities_and_phases(self):
        prob = np.abs(self.psi)
        prob **= 2
        return [prob, self.psi]

    def apply_gate(self, gate, gate_arr):
        apply_gate(
            psi=self.psi,
            gate_arr=gate_arr,
            target=gate.target,
            qbit_indices_steps=self.qbit_indices_steps,
            numQubits=self.circ.numQubits,
        )
        # i = 0
        # step_ind = self.qbit_indices_steps[self.circ.numQubits - 1 - gate.target]
        # while i < self.psi.shape[0]:
        #     j = 0
        #     while j < step_ind:
        #         alpha = self.psi[i + j][0]
        #         beta = self.psi[i + j + step_ind][0]
        #         self.psi[i + j] = [alpha * gate_arr[0][0] + beta * gate_arr[0][1]]
        #         self.psi[i + j + step_ind] = [
        #             alpha * gate_arr[1][0] + beta * gate_arr[1][1]
        #         ]
        #         j += 1
        #     i += 2 * step_ind

    """def apply_error_naive(self):
        for i in range(self.circ.numQubits):
            if self.noise["bitflip"] and (np.random.rand() < self.noise["bitflip"]):
                self.apply_to_statevector(
                    csr_array(np.array([[0, 1], [1, 0]], dtype=np.complex128)),
                    i,
                )
            if self.noise["phaseflip"] and (np.random.rand() < self.noise["phaseflip"]):
                self.apply_to_statevector(csr_array(np.array([[1, 0], [0, -1]])), i)"""

    def x(self, gate):
        self.apply_gate(gate, X_GATE)

    def y(self, gate):
        self.apply_gate(gate, Y_GATE)

    def z(self, gate):
        self.apply_gate(gate, Z_GATE)

    def h(self, gate):
        self.apply_gate(gate, H_GATE)

    def cx(self, gate):
        apply_gate_cx(
            psi=self.psi,
            gate_arr=X_GATE,
            qbit_indices_steps=self.qbit_indices_steps,
            numQubits=self.circ.numQubits,
            control=gate.control[0],
            target=gate.target,
        )

        # gate_arr = X_GATE
        # step_ind = self.qbit_indices_steps[self.circ.numQubits - 1 - gate.control[0]]
        # target_step_ind = self.qbit_indices_steps[self.circ.numQubits - 1 - gate.target]
        # # offset i to only look at '1' entries
        # i = step_ind
        # while i < self.psi.shape[0]:
        #     j = 0
        #     while j < step_ind:
        #         if 0 == int((i + j) / target_step_ind) % 2:
        #             alpha = self.psi[i + j][0]
        #             beta = self.psi[i + j + target_step_ind][0]
        #             self.psi[i + j] = [alpha * gate_arr[0][0] + beta * gate_arr[0][1]]
        #             self.psi[i + j + target_step_ind] = [
        #                 alpha * gate_arr[1][0] + beta * gate_arr[1][1]
        #             ]
        #         j += 1
        #     i += 2 * step_ind

    def ccx(self, gate):
        apply_gate_ccx(
            psi=self.psi,
            gate_arr=X_GATE,
            qbit_indices_steps=self.qbit_indices_steps,
            numQubits=self.circ.numQubits,
            control0=gate.control[0],
            control1=gate.control[1],
            target=gate.target,
        )
        # gate_arr = X_GATE
        # control_step_ind_0 = self.qbit_indices_steps[
        #     self.circ.numQubits - 1 - gate.control[0]
        # ]
        # control_step_ind_1 = self.qbit_indices_steps[
        #     self.circ.numQubits - 1 - gate.control[1]
        # ]
        # target_step_ind = self.qbit_indices_steps[self.circ.numQubits - 1 - gate.target]
        # # offset i to only look at '1' entries
        # i = control_step_ind_0
        # while i < self.psi.shape[0]:
        #     j = 0
        #     while j < control_step_ind_0:
        #         if (
        #             1 == int((i + j) / control_step_ind_1) % 2
        #             and 0 == int((i + j) / target_step_ind) % 2
        #         ):
        #             alpha = self.psi[i + j][0]
        #             beta = self.psi[i + j + target_step_ind][0]
        #             self.psi[i + j] = [alpha * gate_arr[0][0] + beta * gate_arr[0][1]]
        #             self.psi[i + j + target_step_ind] = [
        #                 alpha * gate_arr[1][0] + beta * gate_arr[1][1]
        #             ]
        #             j += 1
        #         j += 1
        #     i += 2 * control_step_ind_0

    def rx(self, gate):
        angle = gate.param[0]
        self.apply_gate(
            gate,
            np.array(
                [
                    [np.cos(angle * 0.5), -1j * np.sin(angle * 0.5)],
                    [-1j * np.sin(angle * 0.5), np.cos(angle * 0.5)],
                ],
                dtype=np.complex128,
            ),
        )

    def ry(self, gate):
        angle = gate.param[0]
        self.apply_gate(
            gate,
            np.array(
                [
                    [np.cos(angle * 0.5), -np.sin(angle * 0.5)],
                    [np.sin(angle * 0.5), np.cos(angle * 0.5)],
                ],
                dtype=np.complex128,
            ),
        )

    def rz(self, gate):
        angle = gate.param[0]
        self.apply_gate(
            gate,
            np.array(
                [
                    [np.cos(angle * 0.5) - 1j * np.sin(angle * 0.5), 0],
                    [0, np.cos(angle * 0.5) + 1j * np.sin(angle * 0.5)],
                ],
                dtype=np.complex128,
            ),
        )

    def sqrt_iswap(self, gate):
        self.psi = SQRT_ISWAP.dot(self.psi)

    def phasedXZ(self, gate):
        x_exponent, z_exponent, axis_phase_exponent = gate.param
        self.apply_gate(
            gate,
            np.array(
                [
                    [
                        np.exp(1j * np.pi * x_exponent / 2)
                        * np.cos(np.pi * x_exponent / 2),
                        -1j
                        * np.exp(1j * np.pi * (x_exponent / 2 - axis_phase_exponent))
                        * np.sin(np.pi * x_exponent / 2),
                    ],
                    [
                        -1j
                        * np.exp(
                            1j
                            * np.pi
                            * (x_exponent / 2 + axis_phase_exponent + z_exponent)
                        )
                        * np.sin(np.pi * x_exponent / 2),
                        np.exp(1j * np.pi * (x_exponent / 2 + z_exponent))
                        * np.cos(np.pi * x_exponent / 2),
                    ],
                ],
                dtype=np.complex128,
            ),
        )

    def t(self, gate):
        self.apply_gate(gate, T_GATE)

    def tdg(self, gate):
        self.apply_gate(gate, TDG_GATE)

    def s(self, gate):
        self.apply_gate(gate, S_GATE)

    def sdg(self, gate):
        self.apply_gate(gate, SDG_GATE)

    def op_to_matrix(self, operations):
        matrix = None
        for op in operations:
            if matrix is None:
                matrix = op
            else:
                matrix = kron(matrix, op, format="csr")

        return matrix

    def measure(self, gate):
        gate.target = np.array(gate.target)
        # Reset Unitary Matrix

        measure_zero = csr_array(np.array([[1, 0], [0, 0]]))
        measure_one = csr_array(np.array([[0, 0], [0, 1]]))

        measure_op_zero = np.array([identity(2) for _ in range(self.circ.numQubits)])
        measure_op_zero[(self.circ.numQubits - 1) - gate.target] = measure_zero

        measure_op_one = np.array([identity(2) for _ in range(self.circ.numQubits)])
        measure_op_one[(self.circ.numQubits - 1) - gate.target] = measure_one

        zero_prob_matrix = self.op_to_matrix(measure_op_zero)

        one_prob_matrix = self.op_to_matrix(measure_op_one)

        prob_one = (
            self.psi.conj().T @ one_prob_matrix.conj().T @ one_prob_matrix @ self.psi
        ).real[0, 0]
        prob_zero = (
            self.psi.conj().T @ zero_prob_matrix.conj().T @ zero_prob_matrix @ self.psi
        ).real[0, 0]

        # Coin Flip biased by probability one
        # random == 1 means we measure a 1
        # random == 0 means we measure a 0
        random = np.random.binomial(1, prob_one)
        if random == 0:
            self.psi = zero_prob_matrix / np.sqrt(prob_zero) @ self.psi
        else:
            self.psi = one_prob_matrix / np.sqrt(prob_one) @ self.psi
