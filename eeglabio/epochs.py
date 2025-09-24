import numpy as np
from numpy.core.records import fromarrays
from scipy.io import savemat

from .utils import cart_to_eeglab, logger


def export_set(fname, data, sfreq, events, tmin, tmax, ch_names, event_id=None,
               ch_locs=None, annotations=None, ref_channels="common",
               precision="single", *, epoch_indices=None):
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
    precision : "single" or "double"
        Precision of the exported data (specifically EEG.data in EEGLAB)

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

    if precision not in ("single", "double"):
        raise ValueError(f"Unsupported precision '{precision}', "
                         f"supported precisions are 'single' and 'double'.")
    data = data.astype(precision)

    ch_cnt, epoch_len, trials = data.shape

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
    ev_types = np.array(ev_types)

    # EEGLAB latency, in units of data sample points
    # ensure same int type (int64) as duration
    ev_lat = events[:, 0].astype(np.int64) + 1  # +1 for eeglab indexing

    # event durations should all be 0 except boundaries which we don't have
    ev_dur = np.zeros_like(ev_lat, dtype=np.int64)

    # indices of epochs each event belongs to
    ev_epoch = ev_lat // epoch_len + 1
    if epoch_indices is not None:
        # full-slice assigment to ensure epoch_indices.shape == ev_lat.shape
        ev_epoch[:] = epoch_indices
    if len(ev_epoch) > 0 and max(ev_epoch) > trials:
        # probably due to shifted/wrong events latency
        # reset events to start at the beginning of each epoch
        ev_epoch = np.arange(1, trials + 1, dtype=np.int64)
        ev_lat = (ev_epoch - 1) * epoch_len
        logger.warning("Invalid event latencies, ignored for export.")

    # merge annotations into events array
    if annotations is not None:
        data_len = epoch_len * trials
        annot_lat = np.array(annotations[1]) * sfreq + 1  # +1 for eeglab
        valid_lat_mask = annot_lat <= data_len
        if not np.all(valid_lat_mask):
            # at least some annotations have invalid onsets, discardd
            logger.warning("Some or all annotations have invalid onsets, "
                           "discarded for export.")

        annot_lat = annot_lat[valid_lat_mask]
        annot_types = np.array(annotations[0])[valid_lat_mask]
        annot_dur = np.array(annotations[2])[valid_lat_mask] * sfreq
        # epoch number = sample / epoch len + 1
        annot_epoch = (annot_lat - 1) // epoch_len + 1  # -1 switch back

        all_types = np.append(ev_types, annot_types)
        all_lat = np.append(ev_lat, annot_lat)
        all_dur = np.append(ev_dur, annot_dur)
        all_epoch = np.append(ev_epoch, annot_epoch)
    else:
        all_types = ev_types
        all_lat = ev_lat
        all_dur = ev_dur
        all_epoch = ev_epoch

    # check there's at least one event per epoch
    uniq_epochs = np.unique(all_epoch)
    if epoch_indices is None:
        required_epochs = np.arange(1, trials + 1)
    else:
        required_epochs = epoch_indices
    if not np.array_equal(uniq_epochs, required_epochs):
        # doesn't meet the requirement of at least one event per epoch
        # add dummy events to satisfy this
        logger.warning("Events doesn't meet the requirement of at least one "
                       "event per epoch, adding dummy events")
        missing_mask = np.isin(required_epochs, uniq_epochs,
                               assume_unique=True, invert=True)
        missing_epochs = required_epochs[missing_mask]
        all_types = np.append(all_types, np.full(len(missing_epochs), "dummy"))
        # set dummy events to start at the beginning of each epoch
        all_lat = np.append(all_lat, (missing_epochs - 1) * epoch_len)
        all_dur = np.append(all_dur, np.zeros_like(missing_epochs))
        all_epoch = np.append(all_epoch, missing_epochs)

    # sort based on latency
    order = all_lat.argsort()
    all_types = all_types[order]
    all_lat = all_lat[order]
    all_dur = all_dur[order]
    all_epoch = all_epoch[order]

    # EEGLAB events format, also used for distinguishing epochs/trials
    events = fromarrays([all_types, all_lat, all_dur, all_epoch],
                        names=["type", "latency", "duration", "epoch"])

    # construct epochs array
    # true epochs array, one subarray per events in epoch
    # make sure epoch count is increasing (it should be)
    # splitting code from https://stackoverflow.com/a/43094244/8170714
    epoch_start_idx = np.unique(all_epoch, return_index=True)[1][1:]  # skip 0
    ep_event = np.split(np.arange(1, len(all_epoch) + 1, dtype=np.double),
                        epoch_start_idx)
    # starting latency for each epoch in seconds
    ep_lat_offset = (all_epoch - 1) * epoch_len / sfreq
    all_lat_shifted = all_lat / sfreq - ep_lat_offset  # shifted rel to epoch
    # convert lat, pos, type to cell arrays by converting to object arrays
    ep_lat = np.split(all_lat_shifted.astype(dtype=object) * 1000,
                      epoch_start_idx)
    ep_pos = np.split(all_epoch.astype(dtype=object), epoch_start_idx)
    ep_types = np.split(all_types.astype(dtype=object), epoch_start_idx)

    # regular one event per epoch
    # same as the indices for event epoch, except use array
    # ep_event = [np.array(n) for n in ev_epoch]
    # ep_lat = [np.array(n) for n in ev_lat]
    # ep_types = [np.array(n) for n in ev_types]

    field_names = ["event", "eventlatency", "eventposition", "eventtype"]
    epochs = fromarrays([np.array(arr, dtype=object) for arr in
                         [ep_event, ep_lat, ep_pos, ep_types]],
                        names=field_names)

    if isinstance(ref_channels, list):
        ref_channels = " ".join(ref_channels)

    eeg_d = dict(data=data,
                 setname=fname,
                 nbchan=data.shape[0],
                 pnts=float(epoch_len),
                 trials=float(trials),
                 srate=float(sfreq),
                 xmin=float(tmin),
                 xmax=float(tmax),
                 ref=ref_channels,
                 chanlocs=chanlocs,
                 event=events,
                 epoch=epochs,
                 icawinv=[],
                 icasphere=[],
                 icaweights=[])
    savemat(str(fname), eeg_d, appendmat=False)
