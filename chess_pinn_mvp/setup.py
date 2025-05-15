from setuptools import setup, find_packages

setup(
    name="chess_pinn_mvp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "beautifulsoup4>=4.9.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
    ],
    entry_points={
        'console_scripts': [
            'chess-pinn-analyze=chess_pinn_mvp.main:main',
            'chess-pinn-demo=chess_pinn_mvp.simple_demo:main',
        ],
    },
    author="Oscar Petrov",
    author_email="your.email@example.com",
    description="Physics-Informed Neural Networks for Chess Rating Forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Chess-PINNs-Forecasting",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.7",
)
