# ruff: noqa: E721
from __future__ import annotations

from pathlib import Path

import boost_histogram as bh
import numpy as np

import uhi_serialization as s


def one_D_test_init(storage_type: str):
    h = None
    if storage_type == "weighted":
        h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())
        h.fill([0.3, 0.3, 0.4, 1.2])
    elif storage_type == "weighted_mean":
        h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.WeightedMean())
        h.fill(
            [0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4], weight=[1, 1, 1, 1, 2]
        )
    elif storage_type == "mean":
        h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Mean())
        h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4])
    return {"test_hist": h}


def test_weighted_storage_write():
    s.write_hdf5_schema("test_weighted_storage.h5", one_D_test_init("weighted"))


def test_weighted_storge_read():
    h_init = one_D_test_init("weighted")
    h_constructed = s.read_hdf5_schema(Path("./test_weighted_storage.h5"))

    assert h_init.keys() == h_constructed.keys()

    actual_hist = h_init["test_hist"]
    re_constructed_hist = h_constructed["test_hist"]

    # checking types of the reconstructed axes
    assert type(actual_hist.axes[0]) == type(re_constructed_hist.axes[0])
    assert actual_hist.storage_type == re_constructed_hist.storage_type
    # checking values of the essential inputs of the axes
    assert actual_hist.axes[0].traits == re_constructed_hist.axes[0].traits
    assert np.allclose(
        actual_hist.axes[0].centers,
        re_constructed_hist.axes[0].centers,
        atol=1e-4,
        rtol=1e-9,
    )
    # checking storage values
    assert np.allclose(
        actual_hist.values(), re_constructed_hist.values(), atol=1e-4, rtol=1e-9
    )
    # checking variance variances
    variances = re_constructed_hist.variances()
    assert variances is not None
    assert np.allclose(actual_hist.variances(), variances, atol=1e-4, rtol=1e-9)


def test_weighted_mean_storage_write():
    s.write_hdf5_schema(
        "test_weighted_mean_storage.h5", one_D_test_init("weighted_mean")
    )


def test_weighted_mean_storage_read():
    h_init = one_D_test_init("weighted_mean")
    h_constructed = s.read_hdf5_schema(Path("./test_weighted_mean_storage.h5"))

    assert h_init.keys() == h_constructed.keys()

    actual_hist = h_init["test_hist"]
    re_constructed_hist = h_constructed["test_hist"]

    # checking types of the reconstructed axes
    assert type(actual_hist.axes[0]) == type(re_constructed_hist.axes[0])
    assert actual_hist.storage_type == re_constructed_hist.storage_type
    # checking values of the essential inputs of the axes
    assert actual_hist.axes[0].traits == re_constructed_hist.axes[0].traits
    assert np.allclose(
        actual_hist.axes[0].centers,
        re_constructed_hist.axes[0].centers,
        atol=1e-4,
        rtol=1e-9,
    )
    # checking storage values
    assert np.allclose(
        actual_hist.values(), re_constructed_hist.values(), atol=1e-4, rtol=1e-9
    )
    # checking variance variances
    print(actual_hist.view(), re_constructed_hist.view())
    print(actual_hist.variances())
    # assert np.allclose(actual_hist.variances(), re_constructed_hist.variances(), atol=1e-4, rtol=1e-9)


def test_mean_storage_write():
    s.write_hdf5_schema("test_mean_storage.h5", one_D_test_init("mean"))


def test_mean_storage_read():
    h_init = one_D_test_init("mean")
    h_constructed = s.read_hdf5_schema(Path("./test_mean_storage.h5"))

    assert h_init.keys() == h_constructed.keys()

    actual_hist = h_init["test_hist"]
    re_constructed_hist = h_constructed["test_hist"]

    # checking types of the reconstructed axes
    assert type(actual_hist.axes[0]) == type(re_constructed_hist.axes[0])
    assert actual_hist.storage_type == re_constructed_hist.storage_type
    # checking values of the essential inputs of the axes
    assert actual_hist.axes[0].traits == re_constructed_hist.axes[0].traits
    assert np.allclose(
        actual_hist.axes[0].centers,
        re_constructed_hist.axes[0].centers,
        atol=1e-4,
        rtol=1e-9,
    )
    # checking storage values
    assert np.allclose(
        actual_hist.values(), re_constructed_hist.values(), atol=1e-4, rtol=1e-9
    )
    # checking variance variances
    # assert np.allclose(actual_hist.variances(), re_constructed_hist.variances(), atol=1e-4, rtol=1e-9)
    assert np.allclose(
        actual_hist.counts(), re_constructed_hist.counts(), atol=1e-4, rtol=1e-9
    )
