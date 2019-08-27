from setuptools.command.build_ext import build_ext as _build_ext
from setuptools import setup, find_packages, Command, Extension
import os
import io
import sys
from shutil import rmtree


# Package meta-data.
NAME = "pychangcooper"
DESCRIPTION = "A generic chang and cooper solver for fokker-planck equations"
URL = "https://github.com/grburgess/pychangcooer"
EMAIL = "jmichaelburgess@gmail.com"
AUTHOR = "J. Michael Burgess"
REQUIRES_PYTHON = ">=3.5.0"
VERSION = None

REQUIRED = ["numpy", "scipy", "ipython", "matplotlib", "numba", "pygsl"]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds...")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution...")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine...")
        os.system("twine upload dist/*")

        self.status("Pushing git tags...")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# extra_files = find_data_files("popsynth/data")


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    #   package_data={"": extra_files},
    license="GPL",
    cmdclass={"upload": UploadCommand},
)
