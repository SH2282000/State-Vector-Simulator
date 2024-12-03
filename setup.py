from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_options = {
    "compiler_directives": {"profile": True, "embedsignature": True},
    "annotate": False,
    "gdb_debug": False,
}

extensions = [
    Extension(
        name="QCP.template.apply_gate",
        sources=["QCP/template/apply_gate.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c99"],
        extra_link_args=["-std=c99"],
        language="c",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        **ext_options,
    ),
    include_dirs=[np.get_include()],
)
