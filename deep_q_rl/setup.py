from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Shift utility',
  ext_modules = cythonize("shift.pyx"),
)

