from pathlib import Path

import numpy as np
from astropy.io import fits


GTBIN_FILE = Path(
    "/d6/CAC/sbhattacharyya/Documents/data/"
    "fermi16-yrs/lat_selected_30M_100M_z80_gtbin.fits"
)


with fits.open(GTBIN_FILE, checksum=True, memmap=True) as hdul:
    print("\nHDU structure")
    print("=" * 70)
    hdul.info()

    print("\nExtension names:")
    print([hdu.name for hdu in hdul])

    for index, hdu in enumerate(hdul):
        print(f"\nHDU {index}: {hdu.name}")
        print("-" * 70)

        for key in [
            "PIXTYPE",
            "ORDERING",
            "NSIDE",
            "ORDER",
            "COORDSYS",
            "FIRSTPIX",
            "LASTPIX",
            "INDXSCHM",
            "TSTART",
            "TSTOP",
            "EMIN",
            "EMAX",
        ]:
            if key in hdu.header:
                print(f"{key:10s} = {hdu.header[key]}")

        if isinstance(hdu, fits.BinTableHDU):
            print("Columns:", hdu.columns.names)
            print("Rows   :", len(hdu.data))
