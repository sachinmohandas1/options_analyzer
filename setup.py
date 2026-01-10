"""Setup script for options analyzer."""

from setuptools import setup, find_packages

setup(
    name="options-analyzer",
    version="1.0.0",
    description="Options chain analyzer for premium selling strategies",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.40",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "py_vollib>=1.0.1",
        "scipy>=1.11.0",
        "plotly>=5.18.0",
        "rich>=13.7.0",
        "pydantic>=2.5.0",
    ],
    extras_require={
        "fast": ["py_vollib_vectorized>=0.1.1", "numba>=0.58.0"],
    },
    entry_points={
        "console_scripts": [
            "options-analyzer=main:main",
        ],
    },
)
