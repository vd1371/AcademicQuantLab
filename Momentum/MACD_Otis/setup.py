from setuptools import setup, find_packages

setup(
    name="trading_strategy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'yfinance',
        'seaborn',
        'pytest',
        'pytest-mock'
    ],
    python_requires='>=3.8',
) 