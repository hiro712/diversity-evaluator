
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diversity-evaluator",
    version="0.1.1",
    author="Hiroto ABE",
    author_email="abe.hiroto.t7@dc.tohoku.ac.jp",
    description="A library to compute diversity of QUBO solutions using GED-based scoring.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hiro712/diversity-evaluator",  # リポジトリ URL に置き換えてください
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "networkx",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)