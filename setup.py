"""
Setup script for ReVa AI Vaccine Design Library
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "ReVa AI Vaccine Design Library"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="reva-ai-vaccine-design",
    version="1.0.0",
    author="ReVa AI Team",
    author_email="contact@reva-ai.com",
    description="A comprehensive library for vaccine design using classical and quantum machine learning approaches",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/reva-ai/vaccine-design",
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
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "quantum": [
            "qiskit>=0.40.0",
            "qiskit-machine-learning>=0.5.0",
        ],
    },
    include_package_data=True,
    package_data={
        "reva_lib": [
            "asset/**/*",
            "asset/data/*.csv",
            "asset/data/*.fasta",
            "asset/model/*.h5",
            "asset/model/*.pkl",
            "asset/model/*.joblib",
            "asset/model/Quantum Model/*",
            "asset/label/*.json",
            "asset/json/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "reva=reva_lib.cli:main",
        ],
    },
    keywords=[
        "vaccine design",
        "bioinformatics",
        "machine learning",
        "quantum computing",
        "epitope prediction",
        "protein analysis",
        "molecular docking",
        "immunoinformatics",
    ],
    project_urls={
        "Bug Reports": "https://github.com/reva-ai/vaccine-design/issues",
        "Source": "https://github.com/reva-ai/vaccine-design",
        "Documentation": "https://reva-ai.github.io/vaccine-design/",
    },
) 