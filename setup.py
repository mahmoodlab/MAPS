from setuptools import setup, find_packages

PYPI_REQUIREMENTS = [
    "torch==1.13.1",
    "numpy>=1.21.5",
    "pandas>=1.3.5",
    "scikit_learn>=1.2.1",
    "mlxtend>=0.22.0",
]
setup(
    name='maps',  # A string containing the package’s name.
    version='1.0.0',  # A string containing the package’s version number.
    description='Machine learning for Analysis of Proteomics in Spatial biology',  # A single-line text explaining the package.
    long_description='',  # A string containing a more detailed description of the package.

    maintainer='',  # It's a string providing the current maintainer’s name, if not the author.
    url='https://github.com/mahmoodlab/MAPS/',  # A string providing the package’s homepage URL (usually the GitHub repository or the PyPI page).
    download_url='https://github.com/mahmoodlab/MAPS/',  # A string containing the URL where the package may be downloaded.
    author='Muhammad Shaban',  # A string identifying the package’s creator/author.
    author_email='muhammadshaban.cs@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],

    packages= find_packages('.'),

    # A string list containing only the dependencies necessary for the package to function effectively.
    install_requires=PYPI_REQUIREMENTS,
)
