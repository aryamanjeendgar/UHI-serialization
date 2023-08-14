import h5py
import boost_histogram as bh
from pathlib import Path
import ast
import numpy as np

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
            """Iterating through the axes, calling `create_axes_object` for each of them,
            creating references to new groups and appending it to the `items` dataset defined above"""
            current_axis = CONSTS['axes_dict'][str(axis)[: str(axis).index('(')]]
            match current_axis:
                case 'regular':
                    args_dict = {}
                    args_dict['bins'] = len(axis.edges) - 1
                    args_dict['lower'] = axis.edges[0]
                    args_dict['upper'] = axis.edges[-1]
                    args_dict['underflow'] = axis.traits.underflow
                    args_dict['overflow'] = axis.traits.overflow
                    args_dict['circular'] = axis.traits.circular
                    if axis.metadata is not None:
                        args_dict['metadata'] = axis.metadata
                        create_axes_object(current_axis, f, name, i, True, args_dict)
                    else:
                        create_axes_object(current_axis, f, name, i, False, args_dict)
                case 'variable':
                    args_dict = {}
                    args_dict['edges'] = axis.edges
                    args_dict['underflow'] = axis.traits.underflow
                    args_dict['overflow'] = axis.traits.overflow
                    args_dict['circular'] = axis.traits.circular
                    if axis.metadata is not None:
                        args_dict['metadata'] = axis.metadata
                        create_axes_object(current_axis, f, name, i, True, args_dict)
                    else:
                        create_axes_object(current_axis, f, name, i, False, args_dict)
                case 'boolean':
                    args_dict = {}
                    if axis.metadata is not None:
                        args_dict['metadata'] = axis.metadata
                        create_axes_object(current_axis, f, name, i, True, args_dict)
                    else:
                        create_axes_object(current_axis, f, name, i, False, args_dict)
                case 'category_int' | 'category_str':
                    args_dict = {}
                    s = str(axis)
                    args_dict['items'] = np.array(ast.literal_eval(s[s.find('['): s.find(']') + 1]),
                                                  dtype=object)
                    args_dict['flow'] = axis.traits.growth
                    if axis.metadata is not None:
                        args_dict['metadata'] = axis.metadata
                        create_axes_object(current_axis, f, name, i, True, args_dict)
                    else:
                        create_axes_object(current_axis, f, name, i, False, args_dict)
            axes_path = group_prefix + '/ref_storage/{}_{}'.format(current_axis, i)
            f[group_prefix + '/ref_storage'].create_group('{}_{}'.format(current_axis, i))
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
                       axis_num: int, has_metadata: bool, args_dict, *args) -> h5py.File:
    """Helper function for constructing and adding a new axis in the /ref_storage subfolder inside
    /hist_name of the hdf5_ptr file"""
    hist_folder_storage = hdf5_ptr['/{}/ref_storage'.format(hist_name)]
    ref = hist_folder_storage.create_group(
                    'axis_{}'.format(axis_num))
    match axis_type:
        case "regular":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "An evenly spaced set of continuous bins."
            ref.attrs['bins'] = args_dict['bins']
            ref.attrs['lower'] = args_dict['lower']
            ref.attrs['upper'] = args_dict['upper']
            ref.attrs['underflow'] = args_dict['underflow']
            ref.attrs['overflow'] = args_dict['overflow']
            ref.attrs['circular'] = args_dict['circular']
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args_dict['metadata'].items():
                    ref['/metadata'].attrs[key] = value
        case "variable":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A variably spaced set of continuous bins."
            #HACK: requires `Variable` data is passed in as a
            # numpy array
            ref.create_dataset('axis_{}_edges'.format(axis_num),
                               shape=args_dict['edges'].shape, data=args_dict['edges'])
            ref.attrs['underflow'] = args_dict['underflow']
            ref.attrs['overflow'] = args_dict['overflow']
            ref.attrs['circular'] = args_dict['circular']
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args_dict['metadata'].items():
                    ref['/metadata'].attrs[key] = value
        case "boolean":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A simple true/false axis with no flow."
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args_dict['metadata'].items():
                    ref['/metadata'].attrs[key] = value
        case "category_int":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A set of integer categorical bins in any order."
            ref.create_dataset('axis_{}_categories'.format(axis_num),
                               shape=args_dict['items'].shape, data=args_dict['items'])
            ref.attrs['flow'] = args_dict['flow']
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args_dict['metadata'].items():
                    ref['/metadata'].attrs[key] = value
        case "category_str":
            ref.attrs['type'] = axis_type
            ref.attrs['description'] = "A set of string categorical bins."
            #HACK: Assumes that the input is a numpy array of strings
            # This is typically imposed via `dtype=object`
            ref.create_dataset('axis_{}_categories'.format(axis_num),
                               shape=args_dict['items'].shape, data=args_dict['items'])
            ref.attrs['flow'] = args_dict['flow']
            if has_metadata:
                ref.create_group('metadata')
                for (key, value) in args_dict['metadata'].items():
                    ref['/metadata'].attrs[key] = value
    return hdf5_ptr

def read_hdf5_schema(input_file: h5py.File | Path) -> bh.Histogram:
    return bh.Histogram()
