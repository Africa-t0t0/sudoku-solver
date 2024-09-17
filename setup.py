from setuptools import setup, find_packages

setup(
    name='sudoku_solver',
    version='0.1',
    description='Un solucionador de sudoku',
    packages=find_packages(),  # Encuentra automáticamente las subcarpetas con __init__.py
    python_requires='>=3.12',
)