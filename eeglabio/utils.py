from functools import partial
from pathlib import Path
import logging
import sys
import time

import numpy as np

logger = logging.getLogger('eeglabio')
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(pathname)s:%(lineno)d '
                                        'EEGLABIO: %(levelname)s: '
                                        '%(message)s'))
logger.addHandler(_handler)
logger.propagate = False


def _xyz_cart_to_eeglab_sph(x, y, z):
    """Convert Cartesian coordinates to EEGLAB spherical coordinates.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_points, )
        Array of x coordinates
    y : numpy.ndarray, shape (n_points, )
        Array of y coordinates
    z : numpy.ndarray, shape (n_points, )
        Array of z coordinates

    Returns
    -------
    sph_pts : numpy.ndarray, shape (n_points, 7)
        Array containing points in spherical coordinates
        (sph_theta, sph_phi, sph_radius, theta, radius,
        sph_theta_besa, sph_phi_besa)

    See Also
    --------
    https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/convertlocs.m

    https://www.mathworks.com/help/matlab/ref/cart2sph.html
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


def cart_to_eeglab_sph(cart):
    """Convert Cartesian coordinates to EEGLAB spherical coordinates.
    Implementation is based on
    `EEGLAB's convertlocs <https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/convertlocs.m>`_
    and Matlab's `cart2sph <https://www.mathworks.com/help/matlab/ref/cart2sph.html>`_

    Parameters
    ----------
    cart : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    sph_pts : ndarray, shape (n_points, 7)
        Array containing points in spherical coordinates
        (sph_theta, sph_phi, sph_radius, theta, radius,
        sph_theta_besa, sph_phi_besa)

    See Also
    --------
    cart_to_eeglab
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
    cart : numpy.ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    full_coords : numpy.ndarray, shape (n_channels, 10)
        xyz + spherical and polar coords. See :func:`cart_to_eeglab_sph` for
        more detail.

    See Also
    --------
    cart_to_eeglab_sph
    """
    return np.append(cart, cart_to_eeglab_sph(cart), 1)  # hstack


def export_mne_epochs(inst, fname, precision="single", *, fmt="v5"):
    """Export MNE's Epochs instance to EEGLAB's .set format using
    :func:`.epochs.export_set`.

    Parameters
    ----------
    inst : mne.BaseEpochs
        Epochs instance to save
    fname : str
        Name of the export file.
    fmt : "v5" | "v7.3"
        MATLAB file format, see :func:`.epochs.export_set`.
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

    if inst.annotations is not None and len(inst.annotations) > 0:
        annot = [inst.annotations.description, inst.annotations.onset,
                 inst.annotations.duration]
    else:
        annot = None
    export_set(fname, inst.get_data(), inst.info['sfreq'], inst.events,
               inst.tmin, inst.tmax, inst.ch_names, inst.event_id,
               cart_coords, annot, precision=precision, fmt=fmt)


def export_mne_raw(inst, fname, precision="single", *, fmt="v5"):
    """Export MNE's Raw instance to EEGLAB's .set format using
    :func:`.raw.export_set`.

    Parameters
    ----------
    inst : mne.io.BaseRaw
        Raw instance to save.
    fname : str
        Name of the export file.
    fmt : "v5" | "v7.3"
        MATLAB file format, see :func:`.raw.export_set`.
    """
    from .raw import export_set

    # load data first
    inst.load_data()

    # remove extra epoc and STI channels
    chs_drop = [ch for ch in ['epoc'] if ch in inst.ch_names]

    if isinstance(inst.filenames[0], Path):
        notfif = not (inst.filenames[0].suffix.endswith('.fif'))
    elif isinstance(inst.filenames[0], str):
        notfif = not inst.filenames[0].endswith('.fif')
    else:
        # assume not fif
        notfif = True

    if 'STI 014' in inst.ch_names and notfif:
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

    ch_types = inst.get_channel_types()
    annotations = [inst.annotations.description, inst.annotations.onset,
                   inst.annotations.duration]
    export_set(fname, inst.get_data(), inst.info['sfreq'], inst.ch_names,
               cart_coords, annotations, ch_types=ch_types,
               precision=precision, fmt=fmt)


def fname_to_setname(fname):
    """
    Derive a portable EEGLAB setname from an output filename or path.

    Rationale
    ---------
    Users often pass absolute paths to export functions. If that path
    propagates into EEGLAB metadata, the dataset becomes non-portable
    (e.g., EEG.setname shows '/full/path/to/file.set').

    This helper ensures:
      - setname is based only on the basename
      - no directory components
      - no file extension
    """
    return Path(fname).stem



def _to_microvolts(data, precision):
    if precision not in ("single", "double"):
        raise ValueError(f"Unsupported precision '{precision}', "
                         f"supported precisions are 'single' and 'double'.")
    # scale directly into the output dtype to avoid a float64 temporary, in
    # Fortran order so that MATLAB's column-major layout needs no copy
    data = np.asarray(data)
    out = np.empty(data.shape, precision, order="F")
    return np.multiply(data, 1e6, out=out, casting="same_kind")


def _get_savemat(fmt):
    if fmt == "v5":
        from scipy.io import savemat
        return partial(savemat, appendmat=False)
    if fmt == "v7.3":
        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError("h5py is required to export with fmt='v7.3'") \
                from None
        return _savemat_v73
    raise ValueError(f"Unsupported fmt '{fmt}', supported formats are "
                     f"'v5' and 'v7.3'.")


def _savemat_v73(fname, mdict):
    import h5py

    with h5py.File(fname, "w", userblock_size=512) as fid:
        for key, value in mdict.items():
            _write_h5(fid, key, value)
    header = (f"MATLAB 7.3 MAT-file, Platform: {sys.platform}, "
              f"Created on: {time.asctime()} HDF5 schema 1.00 .")
    # version 0x0200 + "IM" endian indicator is what marks the file as v7.3
    header = header.encode().ljust(116) + b" " * 8 + b"\x00\x02IM"
    with open(fname, "r+b") as fid:
        fid.write(header)


def _write_h5(group, name, value):
    import h5py

    if isinstance(value, np.ndarray) and value.dtype.names:
        struct = group.create_group(name)
        struct.attrs["MATLAB_class"] = np.bytes_("struct")
        struct.attrs["MATLAB_fields"] = np.array(
            [np.frombuffer(field.encode(), "S1")
             for field in value.dtype.names],
            dtype=h5py.vlen_dtype("S1"))
        for field in value.dtype.names:
            if value.size == 1:  # MATLAB stores scalar struct fields inline
                _write_h5(struct, field, value[field].item())
            else:
                _write_h5_refs(struct, field, value[field])
        return
    if isinstance(value, np.ndarray) and value.dtype == object:
        _write_h5_refs(group, name, value).attrs["MATLAB_class"] = \
            np.bytes_("cell")
        return
    if isinstance(value, str):
        value = np.frombuffer(value.encode("utf-16-le"), "<u2")
        matlab_class = "char"
    else:
        value = np.asarray(value)
        matlab_class = dict(float64="double", float32="single").get(
            value.dtype.name, value.dtype.name)
    if value.size:
        dataset = group.create_dataset(name, data=np.atleast_2d(value).T)
    else:  # MATLAB stores empties as their (0x0) dimensions
        dataset = group.create_dataset(name, data=np.zeros(2, np.uint64))
        dataset.attrs["MATLAB_empty"] = np.uint8(1)
    dataset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
    if matlab_class == "char":
        dataset.attrs["MATLAB_int_decode"] = np.int64(2)


def _write_h5_refs(group, name, values):
    import h5py

    refs_group = group.file.require_group("#refs#")
    values = np.atleast_2d(values)
    refs = np.empty(values.shape, h5py.ref_dtype)
    for idx, value in np.ndenumerate(values):
        key = f"{group.name}/{name}{list(idx)}".replace("/", ".")
        _write_h5(refs_group, key, value)
        refs[idx] = refs_group[key].ref
    return group.create_dataset(name, data=refs.T)
