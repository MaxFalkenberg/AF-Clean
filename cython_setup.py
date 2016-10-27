from setuptools import setup, Extension, Command
from Cython.Build import cythonize
import numpy as np



setup(name='cpropagate_basic',
  version='1.0',
  ext_modules=[Extension('cpropagate_basic', ['cpropagate_basic.pyx'])],
  include_dirs=[np.get_include()]
  )
