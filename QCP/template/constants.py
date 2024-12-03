import numpy as np

X_GATE = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_GATE = np.array(
    [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
    dtype=np.complex128,
)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
TDG_GATE = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
SDG_GATE = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
SQRT_ISWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
        [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
        [0, 0, 0, 1],
    ],
    dtype=np.complex128,
)
