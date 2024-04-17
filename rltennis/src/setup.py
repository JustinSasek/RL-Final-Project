from setuptools import setup, find_packages

setup(
    name='rl',
    version='0.0.0',
    author='Justin Sasek and Harshal Bharatia',
    description='RL Final Project',
    license="MIT",
    packages=find_packages(include=["."]),
    python_requires="==3.10.13",
    install_requires=[
        "gym",
        "seaborn",
        "gymnasium>=0.26",
        "tqdm",
        "numpy",
        "torch",
        "matplotlib",
        "wandb",
    ],
)