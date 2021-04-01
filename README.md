# eeglabio

Python I/O support for EEGLAB files

### Installation

Install from [PyPI](https://test.pypi.org/project/eeglabio):

```
pip install -i https://test.pypi.org/simple/ eeglabio
```

### Dependencies

eeglabio requires Python >= 3.6 and the following packages:
* [mne](https://github.com/mne-tools/mne-python)
* [numpy](http://numpy.org/)
* [scipy](https://www.scipy.org/)

### Usage

Export from MNE [`Epochs`](https://mne.tools/stable/generated/mne.Epochs.html) to EEGLAB (`.set`):
```python
import mne
from eeglabio.epochs import export_set
epochs = mne.Epochs(...)
export_set(epochs, "file_name.set")
```

Export from MNE [`Raw`](https://mne.tools/stable/generated/mne.io.Raw.html) to EEGLAB (`.set`):
```python
import mne
from eeglabio.raw import export_set
raw = mne.io.read_raw(...)
export_set(raw, "file_name.set")
```
