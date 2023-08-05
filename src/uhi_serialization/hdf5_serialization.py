import h5py
import json
import boost_histogram as bh

fp = open('./histogram.json')
schema = json.load(fp)

class uhi_hdf5_writer():
    _schema: dict
    _histogram: bh.Histogram

    def __init__(self, histogram):
        # self._schema = schema
        self._histogram = histogram

    def write_hdf5_schema(self) -> h5py.File:
        #TODO: grab names of histogram objects
        op_name = 'take_attribute_from_hist'
        f = h5py.File(op_name, 'w')
        # All referenced objects will be stored inside of /storage
        f.create_group('storage')

        """
        `metadata` code start
        """
        f.create_group('metadata')
        f['/metadata'].attrs['description'] = "Arbitrary metadata dictionary."
        for (key, value) in self._histogram.metadata.items():
            f['/metadata'].attrs[key] = value
        """
        `metadata` code end
        """

        """
        `axes` code start
        """
        f.create_group('axes')
        f['/axes'].attrs['type'] = 'array'
        f['/axes'].attrs['description'] = "A list of the axes of the histogram."
        f['/axes'].create_dataset('items', len(self._histogram.axes), dtype=h5py.special_dtype(ref=h5py.Reference))
        axes_dict = {
            "Regular": 'regular_axis',
            "Variable": 'variable_axis',
            "IntCategory": 'category_int_axis',
            "StrCategory": 'category_str_axis',
            "Boolean": 'boolean_axis'
        }
        for i, axis in enumerate(self._histogram.axes):
            tmp_axis = axes_dict[str(axis)[: str(axis).index('(')]]
            axes_path = '/storage/{}_{}'.format(tmp_axis, i)
            f['/storage'].create_group('{}_{}'.format(tmp_axis, i))
        """
        `axes` code end
        """

        """
        `storage` code start
        """
        f.create_group('storage')
        f['/group'].attrs['description'] = "The storage of the bins of the histogram."
        """
        `storage` code end
        """

        return f


class uhi_hdf5_reader():
    _schema: dict
    _hdf5_file: h5py.File

    def __init__(self):
        pass

    def read_hdf5_schema(self) -> bh.Histogram:
        return bh.Histogram()
