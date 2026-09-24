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
    event_samples = (
        np.arange(len(epochs)) * len(epochs.times)
        + epochs.time_as_index(0)[0]
    )
    assert_array_equal(event_samples, epochs_read.events[:, 0])
    assert epochs.event_id.keys() == epochs_read.event_id.keys()  # just keys
    assert_allclose(epochs.times, epochs_read.times)
    assert_allclose(epochs.get_data(), epochs_read.get_data())


@pytest.mark.parametrize(('tmin', 'tmax', 'n_times', 'expected', 'relative'), (
    (-0.1, 0.2, 31, [11, 42, 73], 0),
    (-0.1, 0.0, 11, [11, 22, 33], 0),
    (0.1, 0.2, 11, [1, 12, 23], 100),
    (-0.2, -0.1, 11, [11, 22, 33], -100),
))
@pytest.mark.parametrize('selection', (None, [0, 2, 4], [4, 0, 2]))
def test_export_set_dropped_epochs(
        tmp_path, tmin, tmax, n_times, expected, relative, selection):
    """Map retained MNE events to consecutive EEGLAB epochs."""
    data = np.arange(3 * n_times).reshape(3, 1, n_times) * 1e-6
    events = np.array([[100, 0, 1], [300, 0, 2], [500, 0, 3]])
    if selection == [4, 0, 2]:
        data, events = data[[2, 0, 1]], events[[2, 0, 1]]
    event_id = {f'event-{code}': code for code in events[:, 2]}
    fname = tmp_path / 'dropped.set'

    kwargs = dict(fname=fname, data=data, sfreq=100, events=events,
                  tmin=tmin, tmax=tmax, ch_names=['Cz'], event_id=event_id)
    if selection is not None:
        kwargs['epoch_indices'] = np.array(selection)
    export_set(**kwargs)
    epochs_read = read_epochs_eeglab(fname, verbose='error')

    eeglab = loadmat(fname, squeeze_me=True, struct_as_record=False)
    mat_events = np.atleast_1d(eeglab['event'])
    assert_array_equal([event.latency for event in mat_events], expected)
    assert_array_equal([event.epoch for event in mat_events], [1, 2, 3])
    assert_allclose([epoch.eventlatency for epoch in
                     np.atleast_1d(eeglab['epoch'])], relative, atol=1e-12)

    assert_array_equal(epochs_read.events[:, 0], np.array(expected) - 1)
    assert_array_equal([event.type for event in mat_events],
                       [f'event-{code}' for code in events[:, 2]])
    assert_allclose(epochs_read.get_data(), data, atol=1e-12)
    assert set(epochs_read.event_id) == set(event_id)


def test_export_set_multiple_events(tmp_path):
    """Preserve stimulus/response events in already-concatenated epochs."""
    events = np.array([[2, 0, 1], [8, 0, 2], [15, 0, 3], [29, 0, 4]])
    fname = tmp_path / 'multiple.set'
    kwargs = dict(fname=fname, data=np.zeros((3, 1, 11)), sfreq=100,
                  events=events, tmin=-0.1, tmax=0, ch_names=['Cz'])
    export_set(**kwargs, epoch_indices=np.array([0, 0, 1, 2]))
    eeglab = loadmat(fname, squeeze_me=True, struct_as_record=False)
    mat_events = np.atleast_1d(eeglab['event'])
    assert_array_equal([event.latency for event in mat_events], [3, 9, 16, 30])
    assert_array_equal([event.epoch for event in mat_events], [1, 1, 2, 3])
    for epoch, latencies in zip(eeglab['epoch'], ([-80, -20], [-60], [-30])):
        assert_allclose(np.asarray(epoch.eventlatency, dtype=float),
                        np.squeeze(latencies))
    with pytest.raises(ValueError, match='4 events and 3 epochs'):
        export_set(**kwargs)
    with pytest.raises(ValueError, match=r'got shape \(4, 1\)'):
        export_set(**kwargs, epoch_indices=np.array([[0], [0], [1], [2]]))
    with pytest.raises(ValueError, match='got dtype float64'):
        export_set(**kwargs, epoch_indices=np.array([0., 0., 1., 2.]))
    with pytest.raises(ValueError, match='values from 0 to 3'):
        export_set(**kwargs, epoch_indices=np.array([0, 0, 1, 3]))
    with pytest.raises(ValueError, match='within the mapped epoch'):
        export_set(**kwargs, epoch_indices=np.array([0, 1, 1, 2]))
    with pytest.raises(ValueError, match='non-negative'):
        export_set(**{**kwargs, 'events': events[:3]},
                   epoch_indices=np.array([-3, -2, -1]))


@pytest.mark.parametrize('dtype', ('U', 'O', 'T'))
def test_epoch_annotations(tmp_path, dtype):
    """Keep one epoch record per trial, including equal event counts."""
    if dtype == 'T' and not hasattr(getattr(np, 'dtypes', None),
                                    'StringDType'):
        pytest.skip('StringDType requires NumPy 2')
    fname = tmp_path / 'annotations.set'
    export_set(fname, np.zeros((2, 1, 11)), 100,
               np.array([[100, 0, 1], [300, 0, 2]]), -0.1, 0, ['Cz'],
               annotations=[np.array(['before', 'a', 'b', 'after'], dtype),
                            np.array([-.01, 0., .11, .22]),
                            np.array([0., .02, .05, 0.])])
    saved = loadmat(fname, squeeze_me=True, struct_as_record=False)
    assert saved['epoch'].shape == (2,)
    assert_array_equal([event.epoch for event in saved['event']], [1, 1, 2, 2])
    assert_array_equal([event.type for event in saved['event']],
                       ['a', '1', 'b', '2'])
    assert_allclose([event.duration for event in saved['event']], [2, 0, 5, 0])
    for epoch, references, labels in zip(saved['epoch'],
                                         ([1, 2], [3, 4]),
                                         (['a', '1'], ['b', '2'])):
        assert_array_equal(epoch.event, references)
        assert_array_equal(epoch.eventtype, labels)
        assert_allclose(np.asarray(epoch.eventlatency, float), [-100, 0])
