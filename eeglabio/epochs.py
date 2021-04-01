import numpy as np
from numpy.core.records import fromarrays
from scipy.io import savemat
from utils import _get_eeglab_full_cords


def export_set(inst, fname):
    """Export Epochs to EEGLAB's .set format.

    Parameters
    ----------
    inst : mne.BaseEpochs
        Epochs instance to save
    fname : str
        Name of the export file.

    Notes
    -----
    Channel locations are expanded to the full EEGLAB format
    For more details see .io.utils.cart_to_eeglab_full_coords
    """
    # load data first
    inst.load_data()

    # remove extra epoc and STI channels
    chs_drop = [ch for ch in ['epoc', 'STI 014'] if ch in inst.ch_names]
    inst.drop_channels(chs_drop)

    data = inst.get_data() * 1e6  # convert to microvolts
    data = np.moveaxis(data, 0, 2)  # convert to EEGLAB 3D format
    fs = inst.info["sfreq"]
    times = inst.times
    trials = len(inst.events)  # epoch count in EEGLAB

    # get full EEGLAB coordinates to export
    full_coords = _get_eeglab_full_cords(inst)

    ch_names = inst.ch_names

    # convert to record arrays for MATLAB format
    chanlocs = fromarrays(
        [ch_names, *full_coords.T, np.repeat('', len(ch_names))],
        names=["labels", "X", "Y", "Z", "sph_theta", "sph_phi",
               "sph_radius", "theta", "radius",
               "sph_theta_besa", "sph_phi_besa", "type"])

    # reverse order of event type dict to look up events faster
    event_type_d = dict((v, k) for k, v in inst.event_id.items())
    ev_types = [event_type_d[ev[2]] for ev in inst.events]

    # EEGLAB latency, in units of data sample points
    # ev_lat = [int(n) for n in self.events[:, 0]]
    ev_lat = inst.events[:, 0]

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

    eeg_d = dict(EEG=dict(data=data,
                          setname=fname,
                          nbchan=data.shape[0],
                          pnts=float(data.shape[1]),
                          trials=trials,
                          srate=fs,
                          xmin=times[0],
                          xmax=times[-1],
                          chanlocs=chanlocs,
                          event=events,
                          epoch=epochs,
                          icawinv=[],
                          icasphere=[],
                          icaweights=[]))
    savemat(fname, eeg_d,
            appendmat=False)
