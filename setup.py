from setuptools import setup, find_packages

setup(
    name="symr_fire2",          # This is the package name for pip
    version="0.1.0",
    packages={"symr_fire2": "functions"},  # <- map 'symr' module to 'functions' folder
    install_requires=[
        "numpy",
        "pandas",
        "pysr",
        "matplotlib",
        "seaborn",
        "sympy",
        "functools",
        "xgboost",
        "sklearn",
        "shap",
        "scipy",
        "pickle",
        "astropy",
        "itertools",
        "latex2sympy2",
        ],
    python_requires='>=3.8',
    author="Diane Salim",
    description="scripts to run PySR symbolic regression module on FIRE-2 galaxy simulations",
    url="https://github.com/dmsalim/SYMR_FIRE2_GIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)