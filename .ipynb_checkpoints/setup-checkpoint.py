from setuptools import setup, find_packages

setup(
    name="symr_fire2",  # module/package name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # list your dependencies here, e.g.
        # "numpy", "pandas"
    ],
    python_requires='>=3.8',
)