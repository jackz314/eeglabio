from pathlib import Path
from os import path as op
import numpy as np
from mne.io import read_raw_fif, read_raw_eeglab
from numpy.testing import assert_allclose
from raw import export_set
from utils_tests import _TempDir


def test_export_set():
    """Test saving a Raw instance to EEGLAB's set format"""
    fname = Path(__file__).parent / "data" / "test_raw.fif"
    raw = read_raw_fif(fname)
    raw.load_data()
    tmpdir = _TempDir()
    temp_fname = op.join(str(tmpdir), 'test.set')
    export_set(raw, temp_fname)
    raw_read = read_raw_eeglab(temp_fname, preload=True)
    assert raw.ch_names == raw_read.ch_names
    cart_coords = np.array([d['loc'][:3] for d in raw.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3] for d in raw_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read)
    assert_allclose(raw.times, raw_read.times)
    assert_allclose(raw.get_data(), raw_read.get_data())
