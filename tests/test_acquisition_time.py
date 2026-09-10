import os
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.main import extract_acquisition_time


def test_sentinel1_standard_filename():
    filename = "S1A_IW_GRDH_1SDV_20210512T142030_20210512T142055_037856_047713_4A5B.SAFE.tif"
    result = extract_acquisition_time(filename, "/fake/path/file.tif")
    assert result == "2021-05-12 14:20:30 UTC"


def test_sentinel1_tiled_filename():
    filename = "tile_01_S1B_IW_GRDH_1SDV_20230815T091522_20230815T091547_049400_05D1F2_C89E.tif"
    result = extract_acquisition_time(filename, "/fake/path/tile.tif")
    assert result == "2023-08-15 09:15:22 UTC"


def test_non_sentinel1_filename_no_metadata():
    filename = "sample_optical_image.png"
    result = extract_acquisition_time(filename, "/fake/path/image.png")
    assert result == "Acquisition time unavailable"


def test_empty_or_none_filename():
    assert extract_acquisition_time(None, None) == "Acquisition time unavailable"
    assert extract_acquisition_time("", "") == "Acquisition time unavailable"


def test_geotiff_metadata_fallback():
    # Create a temporary GeoTIFF with TIFFTAG_DATETIME set
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        data = np.zeros((1, 16, 16), dtype=np.uint8)
        transform = from_origin(0, 0, 1, 1)

        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype=data.dtype,
            transform=transform,
        ) as dst:
            dst.write(data)
            dst.update_tags(TIFFTAG_DATETIME="2022:04:19 11:22:33")

        # Filename does not contain timestamp
        generic_filename = "unnamed_geotiff.tif"
        result = extract_acquisition_time(generic_filename, tmp_path)
        assert result == "2022-04-19 11:22:33 UTC"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_geotiff_iso_metadata_fallback():
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        data = np.zeros((1, 16, 16), dtype=np.uint8)
        transform = from_origin(0, 0, 1, 1)

        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype=data.dtype,
            transform=transform,
        ) as dst:
            dst.write(data)
            dst.update_tags(DATETIME="2023-11-05 08:30:00")

        generic_filename = "scene.tif"
        result = extract_acquisition_time(generic_filename, tmp_path)
        assert result == "2023-11-05 08:30:00 UTC"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
