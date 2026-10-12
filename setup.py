from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements():
    """Return non-comment entries from the repository requirements file."""
    return [
        line
        for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]

setup(
    name='sd-dpc',
    version='0.1.0',
    description='Sparse Dictionary Differentiable Predictive Control',
    author='Ali Reza Daneshvar Garmroodi',
    author_email='adanesh6@jh.edu',
    url='https://github.com/SOLARIS-JHU/SD-DPC',
    python_requires='>=3.10',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
)
