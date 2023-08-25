#!/usr/bin/env python3
import h5py
import boost_histogram as bh
from numpy.testing import assert_array_equal
import numpy as np
import uhi_serialization as s
from pathlib import Path

def test_1_write():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())
    h.fill([0.3, 0.3, 0.4, 1.2])

    single_test_hist = {
        'test_hist': h
    }
    s.write_hdf5_schema('test_1.h5', single_test_hist)

def test_1_read():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())
    h.fill([0.3, 0.3, 0.4, 1.2])
    h_init = {
        'test_hist': h
    }

    h_constructed = s.read_hdf5_schema(Path('./test_1.h5'))

    assert h_init.keys() == h_constructed.keys()

def test_setting_weight():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())

    h.fill([0.3, 0.3, 0.4, 1.2])

    assert h[0] == bh.accumulators.WeightedSum(3, 3)
    assert h[1] == bh.accumulators.WeightedSum(1, 1)

    h[0] = bh.accumulators.WeightedSum(value=2, variance=2)
    assert h[0] == bh.accumulators.WeightedSum(2, 2)

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)

    assert b["value"][0] == h[0].value
    assert b["variance"][0] == h[0].variance

    h[0] = bh.accumulators.WeightedSum(value=3, variance=1)

    assert h[0].value == 3
    assert h[0].variance == 1

    assert a[0] == h[0]

    assert b["value"][0] == h[0].value
    assert b["variance"][0] == h[0].variance

    assert b[0]["value"] == a[0]["value"]
    assert b[0]["variance"] == a[0]["variance"]

    assert b["value"][0] == a["value"][0]
    assert b["variance"][0] == a["variance"][0]

    assert_array_equal(a.view().value, b.view()["value"])
    assert_array_equal(a.view().variance, b.view()["variance"])
