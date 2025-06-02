
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setuptools.setup(
    name="diversity-evaluator",
    version="0.2.0",
    author="Hiroto ABE",
    author_email="abe.hiroto.t7@dc.tohoku.ac.jp",
    description="A library to compute diversity of QUBO solutions using GED-based scoring.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hiro712/diversity-evaluator",
    packages=setuptools.find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)