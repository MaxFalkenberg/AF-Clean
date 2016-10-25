from setuptools import setup, Extension, Command
from Cython.Build import cythonize
import numpy as np



setup(name='propagate_cython2',
  version='1.0',
  ext_modules=[Extension('propagate_cython2', ['propagate_cython2.pyx'])],
  include_dirs=[np.get_include()]
  )
