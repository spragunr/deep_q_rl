from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'Shift utility',
  ext_modules = cythonize("shift.pyx"),
  include_dirs=[numpy.get_include()]
)

