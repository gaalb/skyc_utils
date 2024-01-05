from setuptools import setup, find_packages

setup(
    name="skyc_utils",
    version="1.0",
    packages=find_packages(),
    install_requires=["matplotlib",
                      "numpy",
                      "scipy"]
)