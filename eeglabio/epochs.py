import numpy as np
from numpy.core.records import fromarrays
from scipy.io import savemat

from .utils import cart_to_eeglab


def export_set(fname, data, sfreq, events, tmin, tmax, ch_names,
               event_id=None, ch_locs=None):
    """Export epoch data to EEGLAB's .set format.

    Parameters
    ----------
    fname : str
        Name of the export file.
    data : numpy.ndarray, shape (n_epochs, n_channels, n_samples)
        Data array containing epochs. Follows the same format as
        MNE Epochs' data array.
    sfreq : int
        sample frequency of data
    events : numpy.ndarray, shape (n_events, 3)
        Event array, the first column contains the event time in samples,
        the second column contains the value of the stim channel immediately
        before the event/step, and the third column contains the event id.
        Follows the same format as MNE's event arrays.
    tmin : float
        Start time (seconds) before event.
    tmax : float
        End time (seconds) after event.
    ch_names : list of str
        Channel names.
    event_id : dict
        Names of conditions corresponding to event ids (last column of events).
        If None, event names will default to string versions of the event ids.
    ch_locs : numpy.ndarray, shape (n_channels, 3)
        Array containing channel locations in Cartesian coordinates (x, y, z)

    See Also
    --------
    .raw.export_set

    Notes
    -----
    Channel locations are expanded to the full EEGLAB format.
    For more details see :func:`.utils.cart_to_eeglab_sph`.
    """

    data = data * 1e6  # convert to microvolts
    data = np.moveaxis(data, 0, 2)  # convert to EEGLAB 3D format

    trials = len(events)  # epoch count in EEGLAB

    if ch_locs is not None:
        # get full EEGLAB coordinates to export
        full_coords = cart_to_eeglab(ch_locs)

        # convert to record arrays for MATLAB format
        chanlocs = fromarrays(
            [ch_names, *full_coords.T, np.repeat('', len(ch_names))],
            names=["labels", "X", "Y", "Z", "sph_theta", "sph_phi",
                   "sph_radius", "theta", "radius",
                   "sph_theta_besa", "sph_phi_besa", "type"])
    else:
        chanlocs = fromarrays([ch_names], names=["labels"])

    # reverse order of event type dict to look up events faster
    # name: value to value: name
    if event_id:
        event_type_d = dict((v, k) for k, v in event_id.items())
        ev_types = [event_type_d[ev[2]] for ev in events]
    else:
        ev_types = [str(ev[2]) for ev in events]

    # EEGLAB latency, in units of data sample points
    # ev_lat = [int(n) for n in self.events[:, 0]]
    ev_lat = events[:, 0]

    # event durations should all be 0 except boundaries which we don't have
    ev_dur = np.zeros((trials,), dtype=np.int64)

    # indices of epochs each event belongs to
    ev_epoch = np.arange(1, trials + 1)

    # EEGLAB events format, also used for distinguishing epochs/trials
    events = fromarrays([ev_types, ev_lat, ev_dur, ev_epoch],
                        names=["type", "latency", "duration", "epoch"])

    # same as the indices for event epoch, except need to use array
    ep_event = [np.array(n) for n in ev_epoch]
    ep_lat = [np.array(n) for n in ev_lat]
    ep_types = [np.array(n) for n in ev_types]

    epochs = fromarrays([ep_event, ep_lat, ep_types],
                        names=["event", "eventlatency", "eventtype"])

    eeg_d = dict(data=data,
                 setname=fname,
                 nbchan=data.shape[0],
                 pnts=float(data.shape[1]),
                 trials=trials,
                 srate=sfreq,
                 xmin=tmin,
                 xmax=tmax,
                 chanlocs=chanlocs,
                 event=events,
                 epoch=epochs,
                 icawinv=[],
                 icasphere=[],
                 icaweights=[])
    savemat(fname, eeg_d, appendmat=False)
