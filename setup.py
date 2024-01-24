from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fastFMRI",
    version="1.0.0",
    author="Yuhui Zhao",
    author_email="yhzhao343@gmail.com",
    url='https://github.com/yhzhao343/fastFMRI',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
    install_requires=["numpy", "scipy", "numba", "connected-components-3d", "nibabel"],
)
