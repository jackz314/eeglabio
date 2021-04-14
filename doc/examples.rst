Example Usage with `MNE <http://mne.tools/>`_
=============================================

Export from :class:`mne.Epochs` to EEGLAB (``.set``):

    .. code-block:: python

        import mne
        from eeglabio.utils import export_mne_epochs
        epochs = mne.Epochs(...)
        export_mne_epochs(epochs, "file_name.set")

Export from :class:`mne.io.Raw` to EEGLAB (``.set``):

    .. code-block:: python

        import mne
        from eeglabio.utils import export_mne_raw
        raw = mne.io.read_raw(...)
        export_mne_raw(raw, "file_name.set")
