from setuptools import setup, find_packages

setup(
    name='rl_final_project',
    version='0.0.0',
    author='Justin Sasek and Harshal Bharatia',
    description='RL Final Project',
    license="MIT",
    packages=find_packages(include=["rl_final_project"]),
    python_requires="==3.10.13",
    install_requires=[
        "gym",
        "gymnasium>=0.26",
        "tqdm",
        "numpy",
        "torch",
        "matplotlib",
        "wandb",
    ],
)