import h5py
import boost_histogram as bh
from pathlib import Path


def write_hdf5_schema(file_name, histograms: dict[str, bh.Histogram]) -> h5py.File:
    f = h5py.File(file_name, 'w')
    for (name, histogram) in histograms.items():
        # All referenced objects will be stored inside of /{name}/ref_storage
        f.create_group('{}'.format(name))
        group_prefix = '/{}'.format(name)
        f[group_prefix].create_group('ref_storage')

        """
        `metadata` code start
        """
        f[group_prefix].create_group('metadata')
        f[group_prefix + '/metadata'].attrs['description'] = "Arbitrary metadata dictionary."
        if histogram.metadata is not None:
            for (key, value) in histogram.metadata.items():
                f[group_prefix + '/metadata'].attrs[key] = value
        """
        `metadata` code end
        """

        """
        `axes` code start
        """
        f[group_prefix].create_group('axes')
        # f[group_prefix + '/axes'].attrs['type'] = 'array'
        f[group_prefix + '/axes'].attrs['description'] = "A list of the axes of the histogram."
        f[group_prefix + '/axes'].create_dataset('items', len(histogram.axes), dtype=h5py.special_dtype(ref=h5py.Reference))
        axes_dict = {
            "Regular": 'regular_axis',
            "Variable": 'variable_axis',
            "IntCategory": 'category_int_axis',
            "StrCategory": 'category_str_axis',
            "Boolean": 'boolean_axis'
        }
        for i, axis in enumerate(histogram.axes):
            tmp_axis = axes_dict[str(axis)[: str(axis).index('(')]]
            axes_path = group_prefix + '/ref_storage/{}_{}'.format(tmp_axis, i)
            f[group_prefix + '/ref_storage'].create_group('{}_{}'.format(tmp_axis, i))
        """
        `axes` code end
        """

        """
        `storage` code start
        """
        f[group_prefix].create_group('storage')
        f[group_prefix + '/storage'].attrs['description'] = "The storage of the bins of the histogram."
        """
        `storage` code end
        """
    return f


def read_hdf5_schema(input_file: h5py.File | Path) -> bh.Histogram:
    return bh.Histogram()
