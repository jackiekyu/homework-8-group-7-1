# A minimal setup.py file to make a Python project installable.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]


setuptools.setup(
    name             = "cibin",
    version          = "0.0.1",
    author           = "Sana Khader, Gregory King, Jacqueline Yu",
    author_email     = "jacquelinekyu@berkeley.edu",
    description      = "A Python library to calculate confidence interval bounds through a permutation method.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages         = setuptools.find_packages(),
    classifiers       = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires  = '>= 3.7',
    install_requires = requirements,
)