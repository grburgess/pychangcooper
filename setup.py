from setuptools import setup, find_packages

setup(

    name="pychangcooper",

    packages=find_packages(),
    version='v0.1',
    description='A generic chang and cooper solver for fokker-planck equations',
    author='J. Michael Burgess',
    author_email='jmichaelburgess@gmail.com',

    requires=[
        'numpy'
        'matplotlib'
    ],
)
