from setuptools import setup, Extension, Command
from Cython.Build import cythonize
import numpy as np



setup(name='cpropagate',
  version='1.0',
  ext_modules=[Extension('cpropagate', ['cpropagate.pyx'])],
  include_dirs=[np.get_include()]
  )
