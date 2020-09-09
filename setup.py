import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    author="Vin985",
    description=("A library to detect birds using deep learning"),
    keywords="audio",
    # long_description=read('README.md'),
    name="dlbd",
    version="0.1",
    packages=find_packages(),
    package_data={"": ["*.svg", "*.yaml", "*.zip", "*.ico", "*.bat"]},
)
