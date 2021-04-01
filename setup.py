import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = None
with open(os.path.join('pyeeglab', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip("'")
            break
if version is None:
    version = "0.0.1"

with open("requirements.txt") as f:
    requires = f.read().splitlines()

setuptools.setup(
    name="pyeeglab",
    version=version,
    author="Jack Zhang",
    author_email="zhangmengyu10@gmail.com",
    description="Python support for EEGLAB files",
    license="BSD (3-clause)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jackz314/pyeeglab",
    project_urls={
        "Source": "https://github.com/jackz314/pyeeglab",
        "Tracker": "https://github.com/jackz314/pyeeglab/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=requires,
    keywords="EEG MEG MNE EEGLAB",
)
