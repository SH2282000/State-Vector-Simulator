import math

import numpy as np

from QCP.parser.parseQCP import QCPcircuit


class DMtemplate:
    circ = None

    def __init__(
        self,
        circ: QCPcircuit,
        noise: dict = {
            "bitflip": 0,
            "phaseflip": 0,
            "amplitude_damping": (1, 0),
            "depolarization": 0,
        },
    ) -> None:
        self.circ = circ
        self.rho = np.zeros((2**circ.numQubits, 2**circ.numQubits), dtype=complex)
        self.rho[0, 0] = 1
        self.noise = noise  # Noise wird als Dictionary angegeben, für jede Form von Noise ein Eintrag mit Wahrscheinlichkeit, einzutreten

    def iterate_circ(self):
        """Iterate through the circuit and apply the gates.

        Raises
        ------
        ValueError
            If the circuit is None.
        """
        # Apply noise before iterating through the circuit
        if self.circ is None:
            raise ValueError("circ is None")

        for gates in self.circ.gates:
            for gate in gates:
                getattr(self, gate.name)(gate)
            # Apply noise after finishing one layer batch
            if self.noise["bitflip"] > 0:
                self.noise_bitflip()
            if self.noise["phaseflip"] > 0:
                self.noise_phaseflip()
            if self.noise["depolarization"] > 0:
                self.noise_depolarization()

    def simulate(self):
        # Iterate Circuit
        self.iterate_circ()
        return self.rho

    def reset(self):
        self.rho = np.zeros(
            (2**self.circ.numQubits, 2**self.circ.numQubits), dtype=complex
        )
        self.rho[0, 0] = 1

    def get_probabilities_and_phases(self):
        fake_psi = np.diagonal(self.rho).astype(float)
        return [np.abs(fake_psi), np.floor(fake_psi)]

    # Commented out because can be called directly

    # def plot(self, hide_legend=False, hide_phase=False):
    #     plot_density_matrix(self.rho, hide_legend, hide_phase)

    # def plot_histogram(self, amount):
    #     plot_histogram(self, amount)

    def apply_to_rho(self, gate, target):
        op = [np.eye(2) for _ in range(self.circ.numQubits)]
        op[(self.circ.numQubits - 1) - target] = gate
        matrix = None
        for i in op:
            if matrix is None:
                matrix = i
            else:
                matrix = np.kron(matrix, i)
        self.rho = matrix @ self.rho @ np.conj(matrix).T

    def x(self, gate):
        x = np.array([[0, 1], [1, 0]])
        self.apply_to_rho(x, gate.target)

    def y(self, gate):
        y = np.array([[0, -1j], [1j, 0]])
        self.apply_to_rho(y, gate.target)

    def z(self, gate):
        z = np.array([[1, 0], [0, -1]])
        self.apply_to_rho(z, gate.target)

    def h(self, gate):
        h = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        self.apply_to_rho(h, gate.target)

    def rx(self, gate):
        angle = gate.param[0]
        rx = np.array(
            [
                [math.cos(angle * 0.5), -1j * math.sin(angle * 0.5)],
                [-1j * math.sin(angle * 0.5), math.cos(angle * 0.5)],
            ]
        )
        self.apply_to_rho(rx, gate.target)

    def ry(self, gate):
        angle = gate.param[0]
        ry = np.array(
            [
                [math.cos(angle * 0.5), -math.sin(angle * 0.5)],
                [math.sin(angle * 0.5), math.cos(angle * 0.5)],
            ]
        )
        self.apply_to_rho(ry, gate.target)

    def rz(self, gate):
        angle = gate.param[0]
        rz = np.array(
            [
                [math.cos(angle * 0.5) - 1j * math.sin(angle * 0.5), 0],
                [0, math.cos(angle * 0.5) + 1j * math.sin(angle * 0.5)],
            ]
        )
        self.apply_to_rho(rz, gate.target)

    def phasedXZ(self, gate):
        x_exponent, z_exponent, axis_phase_exponent = gate.param
        phasedXZ = np.array(
            [
                [
                    np.e ** (1j * np.pi * x_exponent / 2)
                    * np.cos(np.pi * x_exponent / 2),
                    -1j
                    * np.e ** (1j * np.pi * (x_exponent / 2 - axis_phase_exponent))
                    * np.sin(np.pi * x_exponent / 2),
                ],
                [
                    -1j
                    * np.e
                    ** (
                        1j * np.pi * (x_exponent / 2 + axis_phase_exponent + z_exponent)
                    )
                    * np.sin(np.pi * x_exponent / 2),
                    np.e ** (1j * np.pi * (x_exponent / 2 + z_exponent))
                    * np.cos(np.pi * x_exponent / 2),
                ],
            ]
        )
        self.apply_to_rho(phasedXZ, gate.target)

    def sqrt_iswap(self, gate):
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )
        self.rho = matrix @ self.rho @ np.conj(matrix).T

    def t(self, gate):
        t_matrix = np.array([[1, 0], [0, np.e ** (1j * np.pi / 4)]])
        self.apply_to_rho(t_matrix, gate.target)

    def tdg(self, gate):
        tdg_matrix = np.array([[1, 0], [0, np.e ** (-1j * np.pi / 4)]])
        self.apply_to_rho(tdg_matrix, gate.target)

    def s(self, gate):
        s_matrix = np.array([[1, 0], [0, 1j]])
        self.apply_to_rho(s_matrix, gate.target)

    def sdg(self, gate):
        sdg_matrix = np.array([[1, 0], [0, -1j]])
        self.apply_to_rho(sdg_matrix, gate.target)

    def op_to_matrix(self, operations):
        matrix = None
        for op in operations:
            if matrix is None:
                matrix = op
            else:
                matrix = np.kron(matrix, op)
        return matrix

    def cx(self, gate):
        x = np.array([[0, 1], [1, 0]])
        zero_matrix = np.array([[1, 0], [0, 0]])
        one_matrix = np.array([[0, 0], [0, 1]])

        # CX is not fired
        zero_op = [np.eye(2) for _ in range(self.circ.numQubits)]
        zero_op[(self.circ.numQubits - 1) - gate.control[0]] = zero_matrix

        # CX is fired
        one_op = [np.eye(2) for _ in range(self.circ.numQubits)]
        one_op[(self.circ.numQubits - 1) - gate.control[0]] = one_matrix
        one_op[(self.circ.numQubits - 1) - gate.target] = x

        # Build the complete matrix for case cx does not fire
        non_fire_matrix = self.op_to_matrix(zero_op)

        # Build the complete matrix for case cx does fire
        fire_matrix = self.op_to_matrix(one_op)

        hole_matrix = non_fire_matrix + fire_matrix

        self.rho = hole_matrix @ self.rho @ np.conj(hole_matrix).T

    def measure(self, gate):
        measure_zero = np.array([[1, 0], [0, 0]])
        measure_one = np.array([[0, 0], [0, 1]])

        measure_op_zero = np.array([np.eye(2) for _ in range(self.circ.numQubits)])
        measure_op_zero[(self.circ.numQubits - 1) - gate.target[0]] = measure_zero

        measure_op_one = np.array([np.eye(2) for _ in range(self.circ.numQubits)])
        measure_op_one[(self.circ.numQubits - 1) - gate.target[0]] = measure_one

        zero_prob_matrix = self.op_to_matrix(measure_op_zero)
        one_prob_matrix = self.op_to_matrix(measure_op_one)

        self.rho = (
            one_prob_matrix @ self.rho @ np.conj(one_prob_matrix).T
            + zero_prob_matrix @ self.rho @ np.conj(zero_prob_matrix).T
        )

    def build_multi_qubit_gate(self, target, matrix):
        op = [np.eye(2) for _ in range(self.circ.numQubits)]
        op[(self.circ.numQubits - 1) - target] = matrix
        return self.op_to_matrix(op)

    def apply_error(self, error_gates: list):
        """Apply error gates to the state.

        Parameters
        ----------
        error_gates : list
            All the error gates that should be applied to the state.
        """
        for qubit in list(range(self.circ.numQubits)):
            error_channel = [
                self.build_multi_qubit_gate(qubit, error) for error in error_gates
            ]
            # Apply the error channel to the state and update the state
            self.rho = np.sum(
                # Build the sum of the error channel applied to the state
                np.array(
                    [error @ self.rho @ np.conj(error).T for error in error_channel]
                ),
                axis=0,
            )

    def noise_bitflip(self):
        error_zero = np.sqrt(1 - self.noise["bitflip"]) * np.array([[1, 0], [0, 1]])
        error_one = np.sqrt(self.noise["bitflip"]) * np.array([[0, 1], [1, 0]])

        ## Print probability of influence
        # print("Bitflip of zero: ")
        # print(np.trace(error_zero @ self.rho @ np.conj(error_zero).T))

        # Print probability of influence
        # print("Bitflip of one: ")
        # print(np.trace(error_one @ self.rho @ np.conj(error_one).T))

        # Apply Error Gates
        self.apply_error([error_zero, error_one])

    def noise_phaseflip(self):
        error_zero = np.sqrt(1 - self.noise["phaseflip"]) * np.array([[1, 0], [0, 1]])
        error_one = np.sqrt(self.noise["phaseflip"]) * np.array([[1, 0], [0, -1]])

        # Print probability of influence
        # print("Phaseflip of zero: ")
        # print(np.trace(error_zero @ self.rho @ np.conj(error_zero).T))

        # Print probability of influence
        # print("Phaseflip of one: ")
        # print(np.trace(error_one @ self.rho @ np.conj(error_one).T))

        # Save the new state
        self.apply_error([error_zero, error_one])

    def noise_depolarization(self):
        """Noise depolarization."""
        errors = [
            np.sqrt(1 - self.noise["depolarization"]) * np.array([[1, 0], [0, 1]]),
            np.sqrt(1 / 3 * self.noise["depolarization"]) * np.array([[0, 1], [1, 0]]),
            np.sqrt(1 / 3 * self.noise["depolarization"])
            * np.array([[0, -1j], [1j, 0]]),
            np.sqrt(1 / 3 * self.noise["depolarization"]) * np.array([[1, 0], [0, -1]]),
        ]

        self.apply_error(error_gates=errors)

    def noise_amplitude_damping(self):
        """Noise amplitude damping."""
        q_temperature, gamma = self.noise["amplitude_damping"]
        errors = [
            np.sqrt(q_temperature) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]]),
            np.sqrt(q_temperature) * np.array([[0, np.sqrt(gamma)], [0, 0]]),
            np.sqrt(1 - q_temperature) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]]),
            np.sqrt(1 - q_temperature) * np.array([[0, 0], [np.sqrt(gamma), 0]]),
        ]

        # Print Zero Temperature Matrix
        # print("Zero Temperature Matrix: ")
        # print(np.array([[q_temperature, 0], [0, 1 - q_temperature]]))

        # Apply Error
        self.apply_error(errors)
