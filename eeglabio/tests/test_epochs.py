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
    # assert_array_equal(epochs.events[:, 0],
    #                    epochs_read.events[:, 0])  # latency
    assert epochs.event_id.keys() == epochs_read.event_id.keys()  # just keys
    assert_allclose(epochs.times, epochs_read.times)
    assert_allclose(epochs.get_data(), epochs_read.get_data())


@pytest.mark.parametrize(('tmax', 'n_times'), ((0.2, 31), (0.0, 11)))
def test_export_set_dropped_epochs(tmp_path, tmax, n_times):
    """Map retained MNE events to consecutive EEGLAB epochs."""
    sfreq = 100.0
    n_epochs = 3
    data = np.zeros((n_epochs, 1, n_times))
    events = np.column_stack((
        [100, 300, 500],
        np.zeros(n_epochs, dtype=int),
        [1, 2, 3],
    ))
    event_id = {f'event-{code}': code for code in events[:, 2]}
    fname = tmp_path / 'dropped.set'

    export_set(
        fname=fname,
        data=data,
        sfreq=sfreq,
        events=events,
        tmin=-0.1,
        tmax=tmax,
        ch_names=['Cz'],
        event_id=event_id,
    )
    epochs_read = read_epochs_eeglab(fname, verbose='error')

    zero_sample = int(round(0.1 * sfreq))
    expected = np.arange(n_epochs) * n_times + zero_sample

    eeglab = loadmat(fname, squeeze_me=True, struct_as_record=False)
    mat_events = np.atleast_1d(eeglab['event'])
    assert_array_equal([event.latency for event in mat_events], expected + 1)
    assert_array_equal([event.epoch for event in mat_events], [1, 2, 3])

    assert_array_equal(epochs_read.events[:, 0], expected)
    assert set(epochs_read.event_id) == set(event_id)
