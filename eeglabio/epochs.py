import numpy as np
try:
    from numpy.rec import fromarrays  # NumPy 2.0+
except ImportError:
    from numpy.core.records import fromarrays  # NumPy <2.0

from scipy.io import savemat

from .utils import cart_to_eeglab, fname_to_setname, logger


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
        With one time-locking event per epoch, rows must follow ``data`` order.
        Original recording sample numbers (as in MNE) are replaced by the
        time-zero position in each exported epoch. If time zero lies outside
        the window, the closest sample is used with a warning.
        For an explicit multiple-event mapping (see ``epoch_indices``), the
        first column instead contains 0-based sample positions in the
        concatenated exported epochs, not the original recording.
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
        second array (float) is onset (seconds from the first sample of the
        concatenated exported epochs, not original recording time),
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
    epoch_indices : numpy.ndarray or None
        1D integer array with one entry per event (same length as ``events``).
        Non-negative indices with two supported cases:

        * If there is exactly one event per data epoch and all indices are
          unique, this is an MNE selection array. Its values may be
          non-consecutive or reordered; events and data must already be in
          matching order. Exported trial numbers are always 1 through
          ``n_epochs``, not the original selection plus one. None selects
          the same one-event-per-epoch behavior.
        * Otherwise, each value is the 0-based exported trial containing
          that event, in the range 0 through ``n_epochs - 1``. Event sample
          positions must already refer to the concatenated exported data.
          For example, with 11 samples per trial, positions 2 and 8 may be
          a stimulus and response in trial 0, and position 15 is in trial 1.

        .. versionadded:: 0.1.2

    See Also
    --------
    .raw.export_set

    Notes
    -----
    Channel locations are expanded to the full EEGLAB format.
    For more details see :func:`.utils.cart_to_eeglab_sph`.
    """

    # Extact path stem for EEG.setname
    setname = fname_to_setname(fname)

    data = data * 1e6  # convert to microvolts
    data = np.moveaxis(data, 0, 2)  # convert to EEGLAB 3D format

    if precision not in ("single", "double"):
        raise ValueError(f"Unsupported precision '{precision}', "
                         f"supported precisions are 'single' and 'double'.")
    data = data.astype(precision)

    ch_cnt, epoch_len, trials = data.shape

    if epoch_indices is None and len(events) != trials:
        raise ValueError(
            "The number of events must match the number of epochs in the "
            "data when epoch_indices is not provided, but got "
            f"{len(events)} events and {trials} epochs")

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

    one_event_per_epoch = epoch_indices is None
    if epoch_indices is not None:
        epoch_indices = np.asarray(epoch_indices)
        if epoch_indices.shape != (len(events),):
            raise ValueError(
                "epoch_indices must be a 1D array with one entry per event, "
                f"but got shape {epoch_indices.shape} for {len(events)} "
                "events")
        if not np.issubdtype(epoch_indices.dtype, np.integer):
            raise ValueError(
                "epoch_indices must contain integers, but got dtype "
                f"{epoch_indices.dtype}")
        if (epoch_indices < 0).any():
            raise ValueError("epoch_indices must be non-negative, but got "
                             f"minimum {epoch_indices.min()}")
        one_event_per_epoch = (len(events) == trials
                               and len(np.unique(epoch_indices)) == trials)
    if one_event_per_epoch:
        zero_sample = int(round(-tmin * sfreq))
        if not 0 <= zero_sample < epoch_len:
            logger.warning("The epoch window does not include time zero; "
                           "events will be placed at the closest sample.")
            zero_sample = min(max(zero_sample, 0), epoch_len - 1)
        ev_epoch = np.arange(1, trials + 1, dtype=np.int64)
        ev_lat = (ev_epoch - 1) * epoch_len + zero_sample + 1
    else:
        if (epoch_indices >= trials).any():
            raise ValueError(
                "epoch_indices must be between 0 and "
                f"{trials - 1}, but got values from {epoch_indices.min()} "
                f"to {epoch_indices.max()}")
        ev_epoch = epoch_indices.astype(np.int64) + 1
        ev_lat = events[:, 0].astype(np.int64) + 1
        if np.any((ev_lat - 1) // epoch_len != epoch_indices):
            raise ValueError("Event samples must lie within the mapped epoch "
                             "in the concatenated exported data")

    # event durations should all be 0 except boundaries which we don't have
    ev_dur = np.zeros_like(ev_lat, dtype=np.int64)

    # merge annotations into events array
    if annotations is not None:
        data_len = epoch_len * trials
        annot_lat = np.array(annotations[1]) * sfreq + 1  # +1 for eeglab
        valid_lat_mask = (annot_lat >= 1) & (annot_lat <= data_len)
        if not np.all(valid_lat_mask):
            # at least some annotations have invalid onsets, discardd
            logger.warning("Some or all annotations have invalid onsets, "
                           "discarded for export.")

        annot_lat = annot_lat[valid_lat_mask]
        annot_types = np.asarray(annotations[0], dtype=object)[valid_lat_mask]
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
    required_epochs = np.arange(1, trials + 1)
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
        all_lat = np.append(all_lat, (missing_epochs - 1) * epoch_len + 1)
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
    # Convert concatenated one-based samples to epoch-relative seconds.
    ep_lat_offset = (all_epoch - 1) * epoch_len
    all_lat_shifted = (all_lat - 1 - ep_lat_offset) / sfreq + tmin
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
    epochs = fromarrays([np.fromiter(arr, dtype=object) for arr in
                         [ep_event, ep_lat, ep_pos, ep_types]],
                        names=field_names)

    if isinstance(ref_channels, list):
        ref_channels = " ".join(ref_channels)

    eeg_d = dict(data=data,
                 setname=setname,
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
