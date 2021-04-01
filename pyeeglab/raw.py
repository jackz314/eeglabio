import numpy as np
from numpy.core.records import fromarrays
from scipy.io import savemat
from utils import _get_eeglab_full_cords


def export_set(inst, fname):
    """Export Raw to EEGLAB's .set format.

    Parameters
    ----------
    inst : mne.io.BaseRaw
        Raw instance to save
    fname : str
        Name of the export file.

    Notes
    -----
    Channel locations are expanded to the full EEGLAB format
    For more details see .utils.cart_to_eeglab_full_coords
    """
    # load data first
    inst.load_data()

    # remove extra epoc and STI channels
    chs_drop = [ch for ch in ['epoc'] if ch in inst.ch_names]
    if 'STI 014' in inst.ch_names and \
            not (inst.filenames[0].endswith('.fif')):
        chs_drop.append('STI 014')
    inst.drop_channels(chs_drop)

    data = inst.get_data() * 1e6  # convert to microvolts
    fs = inst.info["sfreq"]
    times = inst.times

    # convert xyz to full eeglab coordinates
    full_coords = _get_eeglab_full_cords(inst)

    ch_names = inst.ch_names

    # convert to record arrays for MATLAB format
    chanlocs = fromarrays(
        [ch_names, *full_coords.T, np.repeat('', len(ch_names))],
        names=["labels", "X", "Y", "Z", "sph_theta", "sph_phi",
               "sph_radius", "theta", "radius",
               "sph_theta_besa", "sph_phi_besa", "type"])

    events = fromarrays([inst.annotations.description,
                         inst.annotations.onset * fs + 1,
                         inst.annotations.duration * fs],
                        names=["type", "latency", "duration"])
    eeg_d = dict(EEG=dict(data=data,
                          setname=fname,
                          nbchan=data.shape[0],
                          pnts=data.shape[1],
                          trials=1,
                          srate=fs,
                          xmin=times[0],
                          xmax=times[-1],
                          chanlocs=chanlocs,
                          event=events,
                          icawinv=[],
                          icasphere=[],
                          icaweights=[]))

    savemat(fname, eeg_d,
            appendmat=False)
