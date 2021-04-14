import numpy as np
from numpy.core.records import fromarrays
from scipy.io import savemat

from .utils import cart_to_eeglab


def export_set(fname, data, sfreq, ch_names, ch_locs=None, annotations=None):
    """Export continuous raw data to EEGLAB's .set format.

    Parameters
    ----------
    fname : str
        Name of the export file.
    data : numpy.ndarray, shape (n_epochs, n_channels, n_samples)
        Data array containing epochs. Follows the same format as
        MNE Epochs' data array.
    sfreq : int
        sample frequency of data
    ch_names : list of str
        Channel names.
    ch_locs : numpy.ndarray, shape (n_channels, 3)
        Array containing channel locations in Cartesian coordinates (x, y, z)
    annotations : list, shape (3, n_annotations)
        List containing three annotation subarrays:
        first array (str) is description,
        second array (float) is onset (starting time in seconds),
        third array (float) is duration (in seconds)
        This roughly follows MNE's Annotations structure.

    See Also
    --------
    .epochs.export_set

    Notes
    -----
    Channel locations are expanded to the full EEGLAB format.
    For more details see :func:`.utils.cart_to_eeglab_sph`.
    """

    data = data * 1e6  # convert to microvolts

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

    eeg_d = dict(data=data, setname=fname, nbchan=data.shape[0],
                 pnts=data.shape[1], trials=1, srate=sfreq, xmin=0,
                 xmax=data.shape[1] / sfreq, chanlocs=chanlocs, icawinv=[],
                 icasphere=[], icaweights=[])

    if annotations is not None:
        events = fromarrays([annotations[0],
                             annotations[1] * sfreq + 1,
                             annotations[2] * sfreq],
                            names=["type", "latency", "duration"])
        eeg_d['event'] = events

    savemat(fname, eeg_d, appendmat=False)
