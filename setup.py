from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'umapflow',
    version = '0.0.1',
    author = 'Lisa Weijler',
    author_email = 'lweijler@cvl.tuwien.ac.at',
    description = 'Applying UMAP clustering to cell data',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages = find_packages(),
    install_requires = [
        'pandas',
        'matplotlib',
        'seaborn'
    ],
)
