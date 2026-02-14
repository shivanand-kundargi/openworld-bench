from setuptools import setup, find_packages

setup(
    name="openworld-bench",
    version="0.1.0",
    description="Cross-Setting Evaluation Benchmark for DA, DG, and CL Methods",
    author="openworld-bench contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "Pillow>=9.5.0",
        "timm>=0.9.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "vis": ["matplotlib", "seaborn", "wandb"],
    },
)
