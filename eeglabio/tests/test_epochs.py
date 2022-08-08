from os import path as op
from pathlib import Path

import numpy as np
import pytest
from mne import read_events, pick_types, Epochs, read_epochs_eeglab
from mne.io import read_raw_fif
from numpy.testing import assert_allclose

from eeglabio.utils import export_mne_epochs

raw_fname = Path(__file__).parent / "data" / "test_raw.fif"
event_name = Path(__file__).parent / "data" / 'test-eve.fif'


@pytest.mark.skip
def _get_data(preload=False):
    """Get data."""
    raw = read_raw_fif(raw_fname, preload=preload, verbose='warning')
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, eeg=True, stim=True,
                       ecg=True, eog=True, include=['STI 014'],
                       exclude='bads')
    return raw, events, picks


@pytest.mark.parametrize('preload', (True, False))
def test_export_set(tmpdir, preload):
    """Test saving an Epochs instance to EEGLAB's set format"""
    raw, events = _get_data()[:2]
    raw.load_data()
    epochs = Epochs(raw, events, preload=preload)
    # annot_onsets = np.random.randint(0, len(epochs) *
    #                                  (epochs.tmax - epochs.tmin), 10)
    # annot_dur = np.zeros_like(annot_onsets)
    # annot_desc = [''.join(random.choices(string.ascii_letters, k=10))
    #               for _ in range(len(annot_onsets))]
    # annot = mne.Annotations(annot_onsets, annot_dur, annot_desc)
    # epochs.set_annotations(annot)
    temp_fname = op.join(str(tmpdir), 'test_epochs.set')
    export_mne_epochs(epochs, temp_fname)
    epochs_read = read_epochs_eeglab(temp_fname)
    assert epochs.ch_names == epochs_read.ch_names
    cart_coords = np.array([d['loc'][:3]
                            for d in epochs.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3]
                                 for d in epochs_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read)
    assert len(epochs) == len(epochs_read)
    assert len(epochs_read.events) == len(epochs)
    # assert_array_equal(epochs.events[:, 0],
    #                    epochs_read.events[:, 0])  # latency
    assert epochs.event_id.keys() == epochs_read.event_id.keys()  # just keys
    assert_allclose(epochs.times, epochs_read.times)
    assert_allclose(epochs.get_data(), epochs_read.get_data())
