import numpy as np


def _xyz_cart_to_eeglab_sph(x, y, z):
    """Convert Cartesian coordinates to EEGLAB spherical coordinates.

    Also see https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/convertlocs.m

    Parameters
    ----------
    x : np.ndarray, shape (n_points, )
        Array of x coordinates
    y : np.ndarray, shape (n_points, )
        Array of y coordinates
    z : np.ndarray, shape (n_points, )
        Array of z coordinates

    Returns
    -------
    sph_pts : np.ndarray, shape (n_points, 7)
        Array containing points in spherical coordinates
        (sph_theta, sph_phi, sph_radius, theta, radius,
         sph_theta_besa, sph_phi_besa)
    """  # noqa: E501

    assert len(x) == len(y) == len(z)
    out = np.empty((len(x), 7))

    # https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/topo2sph.m
    def topo2sph(theta, radius):
        c = np.empty((len(theta),))
        h = np.empty((len(theta),))
        for i, (t, r) in enumerate(zip(theta, radius)):
            if t >= 0:
                h[i] = 90 - t
            else:
                h[i] = -(90 + t)
            if t != 0:
                c[i] = np.sign(t) * 180 * r
            else:
                c[i] = 180 * r
        return c, h

    # cart to sph, see https://www.mathworks.com/help/matlab/ref/cart2sph.html
    th = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(np.square(x) + np.square(y)))
    sph_r = np.sqrt(np.square(x) + np.square(y) + np.square(z))

    # other stuff needed by EEGLAB
    sph_theta = th / np.pi * 180
    sph_phi = phi / np.pi * 180
    sph_radius = sph_r
    theta = -sph_theta
    radius = 0.5 - sph_phi / 180
    sph_theta_besa, sph_phi_besa = topo2sph(theta, radius)

    # ordered based on EEGLAB order
    out[:, 0] = sph_theta
    out[:, 1] = sph_phi
    out[:, 2] = sph_radius
    out[:, 3] = theta
    out[:, 4] = radius
    out[:, 5] = sph_theta_besa
    out[:, 6] = sph_phi_besa

    out = np.nan_to_num(out)
    return out


def _cart_to_eeglab_sph(cart):
    """Convert Cartesian coordinates to EEGLAB spherical coordinates.

    Also see https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/convertlocs.m

    Parameters
    ----------
    cart : np.ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    sph_pts : np.ndarray, shape (n_points, 7)
        Array containing points in spherical coordinates
        (sph_theta, sph_phi, sph_radius, theta, radius,
         sph_theta_besa, sph_phi_besa)
    """  # noqa: E501

    # based on transforms.py's _cart_to_sph()
    assert cart.ndim == 2 and cart.shape[1] == 3
    cart = np.atleast_2d(cart)
    x, y, z = cart.T
    return _xyz_cart_to_eeglab_sph(x, y, z)


def cart_to_eeglab(cart):
    """Convert Cartesian coordinates to EEGLAB full coordinates.

    Parameters
    ----------
    cart : np.ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    full_coords : np.ndarray, shape (n_channels, 10)
        xyz + spherical and polar coords
        see cart_to_eeglab_full_coords for more detail.
    """
    return np.append(cart, _cart_to_eeglab_sph(cart), 1)  # hstack


def export_mne_epochs(inst, fname):
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
    from .epochs import export_set
    # load data first
    inst.load_data()

    # remove extra epoc and STI channels
    chs_drop = [ch for ch in ['epoc', 'STI 014'] if ch in inst.ch_names]
    inst.drop_channels(chs_drop)

    chs = inst.info["chs"]
    cart_coords = np.array([d['loc'][:3] for d in chs])
    if cart_coords.any():  # has coordinates
        # (-y x z) to (x y z)
        cart_coords[:, 0] = -cart_coords[:, 0]  # -y to y
        # swap x (1) and y (0)
        cart_coords[:, [0, 1]] = cart_coords[:, [1, 0]]
    else:
        cart_coords = None

    export_set(fname, inst.get_data(), inst.info['sfreq'], inst.events,
               inst.tmin, inst.tmax, inst.ch_names, inst.event_id,
               cart_coords)


def export_mne_raw(inst, fname):
    """Export Raw to EEGLAB's .set format.

    Parameters
    ----------
    inst : mne.BaseRaw
        Raw instance to save
    fname : str
        Name of the export file.

    Notes
    -----
    Channel locations are expanded to the full EEGLAB format
    For more details see pyeeglab.utils._cart_to_eeglab_full_coords
    """
    from .raw import export_set
    # load data first
    inst.load_data()

    # remove extra epoc and STI channels
    chs_drop = [ch for ch in ['epoc'] if ch in inst.ch_names]
    if 'STI 014' in inst.ch_names and \
            not (inst.filenames[0].endswith('.fif')):
        chs_drop.append('STI 014')
    inst.drop_channels(chs_drop)

    chs = inst.info["chs"]
    cart_coords = np.array([d['loc'][:3] for d in chs])
    if cart_coords.any():  # has coordinates
        # (-y x z) to (x y z)
        cart_coords[:, 0] = -cart_coords[:, 0]  # -y to y
        # swap x (1) and y (0)
        cart_coords[:, [0, 1]] = cart_coords[:, [1, 0]]
    else:
        cart_coords = None

    annotations = [inst.annotations.description, inst.annotations.onset,
                   inst.annotations.duration]
    export_set(fname, inst.get_data(), inst.info['sfreq'],
               inst.ch_names, cart_coords, annotations)
