import numpy as np
from numpy.core.records import fromarrays
from scipy.io import savemat

from .utils import cart_to_eeglab


def export_set(fname, data, sfreq, ch_names, ch_locs=None, annotations=None,
               ref_channels="common", ch_types=None):
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
        first array (str) is description/name,
        second array (float) is onset (starting time in seconds),
        third array (float) is duration (in seconds)
        This roughly follows MNE's Annotations structure.
    ref_channels : list of str | str
        The name(s) of the channel(s) used to construct the reference,
        'average' for average reference, or 'common' (default) when there's no
        specific reference set. Note that this parameter is only used to inform
        EEGLAB of the existing reference, this method will not reference the
        data for you.
    ch_types : list of str | None
        Channel types e.g. ‘EEG’, ‘MEG’, ‘EMG’, ‘ECG’, ‘Events’, ..

    See Also
    --------
    .epochs.export_set

    Notes
    -----
    Channel locations are expanded to the full EEGLAB format.
    For more details see :func:`.utils.cart_to_eeglab_sph`.
    """

    data = data * 1e6  # convert to microvolts

    # channel types
    ch_types = np.array(ch_types) if ch_types is not None \
        else np.repeat('', len(ch_names))

    if ch_locs is not None:
        # get full EEGLAB coordinates to export
        full_coords = cart_to_eeglab(ch_locs)

        # convert to record arrays for MATLAB format
        chanlocs = fromarrays(
            [ch_names, *full_coords.T, ch_types],
            names=["labels", "X", "Y", "Z", "sph_theta", "sph_phi",
                   "sph_radius", "theta", "radius",
                   "sph_theta_besa", "sph_phi_besa", "type"])
    else:
        chanlocs = fromarrays([ch_names, ch_types], names=["labels", "type"])

    if isinstance(ref_channels, list):
        ref_channels = " ".join(ref_channels)

    eeg_d = dict(data=data, setname=fname, nbchan=data.shape[0],
                 pnts=float(data.shape[1]), trials=1, srate=sfreq, xmin=0.0,
                 xmax=float(data.shape[1] / sfreq), ref=ref_channels,
                 chanlocs=chanlocs, icawinv=[], icasphere=[], icaweights=[])

    # convert annotations to events
    if annotations is not None:
        events = fromarrays([annotations[0],
                             annotations[1] * sfreq + 1,
                             annotations[2] * sfreq],
                            names=["type", "latency", "duration"])
        eeg_d['event'] = events

    savemat(str(fname), eeg_d, appendmat=False)
