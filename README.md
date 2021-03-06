# eeglabio [![Documentation Status](https://readthedocs.org/projects/eeglabio/badge/?version=latest)](https://eeglabio.readthedocs.io/en/latest/?badge=latest) [![Test & Publish](https://github.com/jackz314/eeglabio/actions/workflows/test_publish.yml/badge.svg)](https://github.com/jackz314/eeglabio/actions/workflows/test_publish.yml)

I/O support for EEGLAB files in Python.

### Installation

Install from [PyPI](https://pypi.org/project/eeglabio):

```
pip install eeglabio
```

Install from [Test PyPI](https://test.pypi.org/project/eeglabio) (unreleased):

```
pip install -i https://test.pypi.org/simple/ eeglabio
```

Install the latest version of the code from GitHub directly (unstable):

```
pip install https://github.com/jackz314/eeglabio/archive/main.zip
```

### Dependencies

eeglabio requires Python >= 3.6 and the following packages:
* [numpy](http://numpy.org/)
* [scipy](https://www.scipy.org/)

For testing, we also require the following additional packages:
* [mne](https://github.com/mne-tools/mne-python)


### Example Usage (with [MNE](https://github.com/mne-tools/mne-python))

Export from MNE [`Epochs`](https://mne.tools/stable/generated/mne.Epochs.html) to EEGLAB (`.set`):
```python
import mne
from eeglabio.utils import export_mne_epochs
epochs = mne.Epochs(...)
export_mne_epochs(epochs, "file_name.set")
```

Export from MNE [`Raw`](https://mne.tools/stable/generated/mne.io.Raw.html) to EEGLAB (`.set`):
```python
import mne
from eeglabio.utils import export_mne_raw
raw = mne.io.read_raw(...)
export_mne_raw(raw, "file_name.set")
```
