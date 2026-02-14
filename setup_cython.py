"""Build script for Cython-accelerated game engine.

Usage:
    python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    print("Cython is required to build the fast game engine.")
    print("Install it with: pip install cython")
    raise

extensions = [
    Extension(
        "game_fast",
        sources=["game_fast.pyx"],
    ),
]

setup(
    name="oh-hell-game-fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": 3,
        },
    ),
)
