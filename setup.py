#!/usr/bin/env python3
"""
Setup script for the Transformer Immune Cells package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="transformer-immune-cells",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready pipeline for analyzing immune cell data using transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/transformer_immune",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "transformer-immune=run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    keywords="bioinformatics, single-cell, transformer, geneformer, immunology",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/transformer_immune/issues",
        "Source": "https://github.com/yourusername/transformer_immune",
        "Documentation": "https://github.com/yourusername/transformer_immune#readme",
    },
) 