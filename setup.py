from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = 'KDSH',
    version = '0.1',
    author = 'Attention Is All We Need',
    packages = find_packages(),
    install_requires = requirements,
)

