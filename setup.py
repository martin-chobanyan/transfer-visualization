from setuptools import setup, find_packages

setup(
    name='feature_vis',
    version='0.1dev',
    packages=find_packages(),
    install_requires=[
        'kornia',
        'numpy',
        'torch',
        'torchvision',
        'tqdm'
    ]
)
