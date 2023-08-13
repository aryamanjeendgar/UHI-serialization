import h5py
import boost_histogram as bh
from pathlib import Path

CONSTS = {
    'axes_dict': {
            "Regular": 'regular',
            "Variable": 'variable',
            "IntCategory": 'category_int',
            "StrCategory": 'category_str',
            "Boolean": 'boolean'
    },
    'axes_dict_to_props': {
            'regular': {
                "type": { "type": "string", "const": "regular" },
                "lower": { "type": "number", "description": "Lower edge of the axis." },
                "upper": { "type": "number", "description": "Upper edge of the axis." },
                "bins": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of bins in the axis."
                },
                "underflow": {
                    "type": "boolean",
                    "description": "True if there is a bin for underflow."
                },
                "overflow": {
                    "type": "boolean",
                    "description": "True if there is a bin for overflow."
                },
                "circular": {
                    "type": "boolean",
                    "description": "True if the axis wraps around."
                },
                "metadata": {
                    "type": "object",
                    "description": "Arbitrary metadata dictionary."
                }
            },
            'variable': {
                "type": { "type": "string", "const": "variable" },
                "edges": {
                    "oneOf": [
                        {
                        "type": "array",
                        "items": { "type": "number", "minItems": 2, "uniqueItems": True}
                        },
                        {
                        "type": "string",
                        "description": "A path (URI?) to the edges data."
                        }
                    ]
                },
                "underflow": { "type": "boolean" },
                "overflow": { "type": "boolean" },
                "circular": { "type": "boolean" },
                "metadata": {
                    "type": "object",
                    "description": "Arbitrary metadata dictionary."
                }
            },
            'category_str': {
                "type": { "type": "string", "const": "category_str" },
                "categories": {
                    "type": "array",
                    "items": { "type": "string" },
                    "uniqueItems": True
                },
                "flow": {
                    "type": "boolean",
                    "description": "True if flow bin (at the overflow position) present."
                },
                "metadata": {
                    "type": "object",
                    "description": "Arbitrary metadata dictionary."
                }
            },
            'category_int': {
                "type": { "type": "string", "const": "category_int" },
                "categories": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "uniqueItems": True
                },
                "flow": {
                    "type": "boolean",
                    "description": "True if flow bin (at the overflow position) present."
                },
                "metadata": {
                    "type": "object",
                    "description": "Arbitrary metadata dictionary."
                }
            },
            'boolean': {
                "type": { "type": "string", "const": "boolean" },
                "metadata": {
                    "type": "object",
                    "description": "Arbitrary metadata dictionary."
                }
            }
    }
}

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
        f[group_prefix + '/axes'].attrs['description'] = "A list of the axes of the histogram."
        f[group_prefix + '/axes'].create_dataset('items', len(histogram.axes), dtype=h5py.special_dtype(ref=h5py.Reference))
        for i, axis in enumerate(histogram.axes):
            tmp_axis = CONSTS['axes_dict'][str(axis)[: str(axis).index('(')]]
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


def create_axes_object(axis_type: str, hdf5_ptr: h5py.File, hist_name: str,
                       axis_num: int, has_metadata: bool, *args) -> h5py.File:
    """Helper function for constructing and adding a new axis in the /ref_storage subfolder inside
    /hist_name of the hdf5_ptr file"""
    hist_folder_storage = hdf5_ptr['/{}/ref_storage'.format(hist_name)]
    ref = hist_folder_storage.create_group(
                    'axis_{}'.format(axis_num))
    match axis_type:
        case "regular":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "An evenly spaced set of continuous bins."
            ref.attrs['bins'] = args[0]
            ref.attrs['lower'] = args[1]
            ref.attrs['upper'] = args[2]
            ref.attrs['underflow'] = args[3]
            ref.attrs['overflow'] = args[4]
            ref.attrs['circular'] = args[5]
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args[6].items():
                    ref['/metadata'].attrs[key] = value
        case "variable":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A variably spaced set of continuous bins."
            #HACK: requires `Variable` data is passed in as a
            # numpy array
            ref.create_dataset('axis_{}_edges'.format(axis_num),
                               shape=args[0].shape, data=args[0])
            ref.attrs['underflow'] = args[1]
            ref.attrs['overflow'] = args[2]
            ref.attrs['circular'] = args[3]
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args[4].items():
                    ref['/metadata'].attrs[key] = value
        case "boolean":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A simple true/false axis with no flow."
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args[0].items():
                    ref['/metadata'].attrs[key] = value
        case "str_category":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A set of string categorical bins."
            #HACK: Assumes that the input is a numpy array of strings
            # This is typically imposed via `dtype=object`
            ref.create_dataset('axis_{}_categories'.format(axis_num),
                               shape=args[0].shape, data=args[0])
            ref.attrs['flow'] = args[1]
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args[2].items():
                    ref['/metadata'].attrs[key] = value
        case "int_category":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A set of string categorical bins."
            #HACK: Assumes that the input is a numpy array of strings
            # This is typically imposed via `dtype=object`
            ref.create_dataset('axis_{}_categories'.format(axis_num),
                               shape=args[0].shape, data=args[0])
            ref.attrs['flow'] = args[1]
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args[2].items():
                    ref['/metadata'].attrs[key] = value
    return hdf5_ptr

def read_hdf5_schema(input_file: h5py.File | Path) -> bh.Histogram:
    return bh.Histogram()
