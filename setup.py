import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = None
with open(os.path.join('eeglabio', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip("'")
            break
if version is None:
    version = "0.0.1"

with open("requirements.txt") as f:
    requires = f.read().splitlines()

GITHUB_URL = "https://github.com/jackz314/eeglabio"
setuptools.setup(
    name="eeglabio",
    version=version,
    author="Jack Zhang",
    author_email="zhangmengyu10@gmail.com",
    description="I/O support for EEGLAB files in Python",
    license="BSD (3-clause)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=GITHUB_URL,
    download_url=GITHUB_URL,
    project_urls={
        "Source": GITHUB_URL,
        "Tracker": GITHUB_URL + '/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=("*tests",)),
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=requires,
    keywords="EEG MEG MNE EEGLAB",
)
