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


@pytest.mark.parametrize(('tmax', 'n_times'), ((0.2, 31), (0.0, 11)))
@pytest.mark.parametrize('pass_epoch_indices', (False, True))
def test_export_set_dropped_epochs(
        tmp_path, tmax, n_times, pass_epoch_indices):
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

    kwargs = dict(
        fname=fname,
        data=data,
        sfreq=sfreq,
        events=events,
        tmin=-0.1,
        tmax=tmax,
        ch_names=['Cz'],
        event_id=event_id,
    )
    if pass_epoch_indices:
        kwargs['epoch_indices'] = np.array([0, 2, 4])
    export_set(**kwargs)
    epochs_read = read_epochs_eeglab(fname, verbose='error')

    zero_sample = int(round(0.1 * sfreq))
    expected = np.arange(n_epochs) * n_times + zero_sample

    eeglab = loadmat(fname, squeeze_me=True, struct_as_record=False)
    mat_events = np.atleast_1d(eeglab['event'])
    assert_array_equal([event.latency for event in mat_events], expected + 1)
    assert_array_equal([event.epoch for event in mat_events], [1, 2, 3])

    assert_array_equal(epochs_read.events[:, 0], expected)
    assert set(epochs_read.event_id) == set(event_id)


def test_export_set_multiple_events(tmp_path):
    """Preserve multiple events within an epoch."""
    data = np.zeros((3, 1, 11))
    events = np.column_stack((
        [2, 8, 15, 29],
        np.zeros(4, dtype=int),
        [1, 2, 3, 4],
    ))
    fname = tmp_path / 'multiple.set'
    export_set(
        fname=fname,
        data=data,
        sfreq=100.0,
        events=events,
        tmin=-0.1,
        tmax=0.0,
        ch_names=['Cz'],
        epoch_indices=np.array([0, 0, 1, 2]),
    )

    eeglab = loadmat(fname, squeeze_me=True, struct_as_record=False)
    mat_events = np.atleast_1d(eeglab['event'])
    latencies = [event.latency for event in mat_events]
    assert_array_equal(latencies, events[:, 0] + 1)
    assert_array_equal([event.epoch for event in mat_events], [1, 1, 2, 3])


@pytest.mark.parametrize(('tmin', 'tmax', 'expected'), (
    (0.1, 0.2, [1, 12]),
    (-0.2, -0.1, [11, 22]),
))
def test_export_set_without_time_zero(tmp_path, tmin, tmax, expected):
    """Keep event latencies valid when the epoch excludes time zero."""
    fname = tmp_path / 'no-zero.set'
    export_set(
        fname=fname,
        data=np.zeros((2, 1, 11)),
        sfreq=100.0,
        events=np.array([[100, 0, 1], [300, 0, 2]]),
        tmin=tmin,
        tmax=tmax,
        ch_names=['Cz'],
    )

    eeglab = loadmat(fname, squeeze_me=True, struct_as_record=False)
    mat_events = np.atleast_1d(eeglab['event'])
    assert_array_equal([event.latency for event in mat_events], expected)
    assert_array_equal([event.epoch for event in mat_events], [1, 2])


def test_export_set_epoch_index_validation(tmp_path):
    """Report mismatched event and epoch index counts."""
    data = np.zeros((3, 1, 11))
    events = np.column_stack((
        [100, 300, 500],
        np.zeros(3, dtype=int),
        [1, 2, 3],
    ))
    kwargs = dict(
        fname=tmp_path / 'invalid.set',
        data=data,
        sfreq=100.0,
        events=events,
        tmin=-0.1,
        tmax=0.0,
        ch_names=['Cz'],
    )

    with pytest.raises(ValueError, match='2 events and 3 epochs'):
        export_set(**{**kwargs, 'events': events[:2]})
    with pytest.raises(ValueError, match=r'got shape \(3, 1\)'):
        export_set(**kwargs, epoch_indices=np.array([[0], [2], [4]]))
    with pytest.raises(ValueError, match='got dtype float64'):
        export_set(**kwargs, epoch_indices=np.array([0.0, 2.0, 4.0]))
    with pytest.raises(ValueError, match='values from 0 to 3'):
        export_set(**kwargs, epoch_indices=np.array([0, 0, 3]))
