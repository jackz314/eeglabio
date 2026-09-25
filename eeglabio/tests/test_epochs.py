from os import path as op
from pathlib import Path

import numpy as np
import pytest
from mne import read_events, pick_types, Epochs, read_epochs_eeglab
from mne.io import read_raw_fif
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from eeglabio.epochs import export_set
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
    epochs.drop([1, 3])
    # annot_onsets = np.random.randint(0, len(epochs) *
    #                                  (epochs.tmax - epochs.tmin), 10)
    # annot_dur = np.zeros_like(annot_onsets)
    # annot_desc = [''.join(random.choices(string.ascii_letters, k=10))
    #               for _ in range(len(annot_onsets))]
    # annot = mne.Annotations(annot_onsets, annot_dur, annot_desc)
    # epochs.set_annotations(annot)
    temp_fname = op.join(str(tmpdir), 'test_epochs.set')
    export_mne_epochs(epochs, temp_fname)
    epochs_read = read_epochs_eeglab(temp_fname, montage_units='m')
    assert epochs.ch_names == epochs_read.ch_names
    cart_coords = np.array([d['loc'][:3]
                            for d in epochs.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3]
                                 for d in epochs_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read)
    assert len(epochs) == len(epochs_read)
    assert len(epochs_read.events) == len(epochs)
    event_samples = (
        np.arange(len(epochs)) * len(epochs.times)
        + epochs.time_as_index(0)[0]
    )
    assert_array_equal(event_samples, epochs_read.events[:, 0])
    assert epochs.event_id.keys() == epochs_read.event_id.keys()  # just keys
    assert_allclose(epochs.times, epochs_read.times)
    assert_allclose(epochs.get_data(), epochs_read.get_data())


@pytest.mark.parametrize('tmin, latency, epoch_latency', [
    (-0.1, [6, 11, 17, 22], [-50, 0]),
    (0.1, [1, 6, 12, 17], [100, 150]),  # time zero clamped to first sample
])
def test_export_set_latency(tmp_path, tmin, latency, epoch_latency):
    """Test event and annotation latencies in EEGLAB's event structures."""
    fname = tmp_path / 'test.set'
    events = np.array([[100, 0, 1], [300, 0, 2]])
    args = (np.zeros((2, 1, 11)), 100., events, tmin, tmin + 0.1, ['Cz'])
    export_set(fname, *args, annotations=[['a', 'b'], [0.05, 0.16], [0, 0]],
               epoch_indices=np.array([0, 5]))
    saved = loadmat(fname, squeeze_me=True, struct_as_record=False)
    assert_array_equal([ev.latency for ev in saved['event']], latency)
    assert_array_equal([ev.epoch for ev in saved['event']], [1, 1, 2, 2])
    assert saved['epoch'].shape == (2,)
    for epoch in saved['epoch']:
        assert_allclose(epoch.eventlatency.astype(float), epoch_latency,
                        atol=1e-9)
    with pytest.raises(ValueError, match='1 events and 2 epochs'):
        export_set(fname, *args[:2], events[:1], *args[3:])
