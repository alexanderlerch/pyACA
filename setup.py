import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

with open(HERE / "README.md", "r") as fh:
    long_description = fh.read()

setup(name="pyACA",
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="scripts accompanying the book An Introduction to Audio Content Analysis by Alexander Lerch",
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
      ],
      keywords="audio analysis features pitch key extraction music onset beat detection descriptors",
      url="https://github.com/alexanderlerch/pyACA",
      author="Alexander Lerch",
      author_email="info@AudioContentAnalysis.org",
      license="MIT",
      packages=["pyACA"],
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
      ],
      zip_safe=False)
