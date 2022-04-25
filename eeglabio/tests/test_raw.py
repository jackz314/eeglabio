from os import path as op
from pathlib import Path

import pytest
import numpy as np
from mne.io import read_raw_fif, read_raw_eeglab
from numpy.testing import assert_allclose

from eeglabio.utils import export_mne_raw

raw_fname = Path(__file__).parent / "data" / "test_raw.fif"


def test_export_set(tmpdir):
    """Test saving a Raw instance to EEGLAB's set format"""
    raw = read_raw_fif(raw_fname)
    raw.load_data()
    temp_fname = op.join(str(tmpdir), 'test_raw.set')
    export_mne_raw(raw, temp_fname)
    with pytest.warns(RuntimeWarning, match='Not setting positions'):
        raw_read = read_raw_eeglab(temp_fname, preload=True)
    assert raw.ch_names == raw_read.ch_names
    cart_coords = np.array([d['loc'][:3] for d in raw.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3] for d in raw_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read, atol=1e-5)
    assert_allclose(raw.times, raw_read.times, atol=1e-5)
    assert_allclose(raw.get_data(), raw_read.get_data(), atol=1e-11)
