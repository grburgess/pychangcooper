#from setuptools import setup
from distutils.core import setup
setup(

    name="pychangcooper",
    packages=[
        'pychangcooper',
        'pychangcooper/utils',
        'pychangcooper/scenarios',

    ],
    version='v1.1',
    description='A generic chang and cooper solver for fokker-planck equations',
    author='J. Michael Burgess',
    author_email='jmichaelburgess@gmail.com',
    url = 'https://github.com/grburgess/pychangcooper',
    download_url='https://github.com/grburgess/pychangcooper/archive/v1.1.tar.gz',
    requires=[
        'numpy',
        'matplotlib'
    ],
)
