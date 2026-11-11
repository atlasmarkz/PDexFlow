from setuptools import find_packages, setup

setup(
    name="pdexflow",
    version="0.1.0",
    description="PDexFlow: parity-harmonized flow matching for de novo D-peptide design",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "omegaconf",
        "hydra-core",
        "tqdm",
        "easydict",
        "lmdb",
        "joblib",
        "biopython",
    ],
)
