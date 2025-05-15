#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chess-pinns-forecasting",
    version="0.1.0",
    author="Chess-PINNs-Forecasting Team",
    description="Chess rating forecasting using Physics-Informed Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Chess-PINNs-Forecasting",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "deepxde>=1.0.0",
        "requests>=2.25.0",
        "ratelimit>=2.2.1",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "chess-pinn=chess_pinn.cli:cli",
        ],
    },
)
