#!/usr/bin/env python3
import h5py
import json
import boost_histogram as bh

fp = open('./histogram.json')
schema = json.load(fp)

class uhi_hdf5_writer():
    _schema: dict
    _histogram: bh.Histogram

    def __init__(self, schema, histogram):
        self._schema = schema
        self._histogram = histogram

    def write_hdf5_schema(self) -> h5py.File:
        return h5py.File('')


class uhi_hdf5_reader():
    _schema: dict
    _hdf5_file: h5py.File

    def __init__(self):
        pass

    def read_hdf5_schema(self) -> bh.Histogram:
        return bh.Histogram()
