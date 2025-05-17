from setuptools import setup, find_packages

reqs=[
    ]

setup(
    name='PharmaDiff',
    version='0.0.1',
    url=None,
    author='anonymous',
    author_email='',
    description='Pharmacophore-Conditioned Diffusion Model for Ligand-Based De Novo Drug Design',
    packages=find_packages(exclude=["wandb", "archives", "configs"]),
    install_requires=reqs
)
