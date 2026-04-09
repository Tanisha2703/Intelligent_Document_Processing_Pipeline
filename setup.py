"""
Setup script for Multilingual Document Intelligence Pipeline.
Allows the package to be installed with: pip install -e .
"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="multilingual-doc-intel",
    version="1.0.0",
    author="Tanis",
    description="End-to-end multilingual document intelligence pipeline for Hindi and English documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(where=".", include=["src", "src.*"]),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "doc-intel=app:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
