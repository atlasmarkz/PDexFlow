from setuptools import find_packages, setup

setup(
    name="unichiflow",
    version="0.1.0",
    description="UniChiFlow: unified chiral-aware flow matching framework (core scaffold, without ChiLoRA)",
    packages=find_packages(),
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
