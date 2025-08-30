"""Setup script for Cross-Cultural Music Recommendation Research System."""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# Read requirements from requirements.txt
def read_requirements():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="cross-cultural-music-research",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Research-quality music recommendation system for cross-cultural preference analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/cross-cultural-music-research",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0", 
            "black>=23.0.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
        ],
        "research": [
            "papermill>=2.4.0",
            "nbconvert>=7.7.0",
            "altair>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "music-research=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md", "*.txt"],
    },
)