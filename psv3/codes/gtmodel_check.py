from pathlib import Path

import numpy as np
from astropy.io import fits


MODEL_FILE = Path(
    "/d6/CAC/sbhattacharyya/Documents/data/"
    "fermi16-yrs/photon/gtmodel/"
    "psr_1_1G_1T_z105_gtmodel_ccube.fits"
)


with fits.open(MODEL_FILE, memmap=True, checksum=True) as hdul:
    print("\nFITS structure")
    print("=" * 70)
    hdul.info()

    print("\nExtensions:")
    print([hdu.name for hdu in hdul])

    sky = hdul["SKYMAP"]

    print("\nGeometry")
    print("=" * 70)
    for key in [
        "PIXTYPE",
        "ORDERING",
        "NSIDE",
        "ORDER",
        "COORDSYS",
        "FIRSTPIX",
        "LASTPIX",
        "INDXSCHM",
    ]:
        print(f"{key:10s}: {sky.header.get(key)}")

    print("\nColumns:")
    print(sky.columns.names)
    print("Rows:", len(sky.data))

    expected_npix = 12 * int(sky.header["NSIDE"]) ** 2
    print("Expected pixels:", expected_npix)
    print("Pixel count OK :", len(sky.data) == expected_npix)

    print("\nModel-map statistics")
    print("=" * 70)

    total_expected_counts = 0.0

    for name in sky.columns.names:
        values = np.asarray(sky.data[name])

        finite = np.isfinite(values)
        channel_sum = np.sum(values, dtype=np.float64)

        print(
            f"{name}: "
            f"min={np.nanmin(values):.6g}, "
            f"max={np.nanmax(values):.6g}, "
            f"sum={channel_sum:,.6f}, "
            f"NaN/inf={np.count_nonzero(~finite)}, "
            f"negative={np.count_nonzero(values < 0)}, "
            f"nonzero={np.count_nonzero(values)}"
        )

        total_expected_counts += channel_sum

    print(
        f"\nTotal expected pulsar counts: "
        f"{total_expected_counts:,.6f}"
    )
