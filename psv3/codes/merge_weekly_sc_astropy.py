'''
since the spacecraft_merged file always turned out to be corrupted
we had to brute force combine all the weekly files
'''


from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


SPACECRAFT_DIR = Path(
    "/d6/CAC/sbhattacharyya/Documents/data/"
    "fermi16-yrs/spacecraft"
)

FILE_PATTERN = "lat_spacecraft_weekly_w*.fits"

OUTPUT_FILE = (
    SPACECRAFT_DIR
    / "lat_spacecraft_weekly_merged_astropy.fits"
)


def validate_weekly_file(path: Path) -> None:
    """Check that a weekly spacecraft FITS file is readable."""

    with fits.open(path, memmap=True) as hdul:
        extension_names = [hdu.name for hdu in hdul]

        if "SC_DATA" not in extension_names:
            raise RuntimeError(
                f"{path.name}: SC_DATA extension missing; "
                f"found {extension_names}"
            )

        columns = set(hdul["SC_DATA"].columns.names)

        required_columns = {
            "START",
            "STOP",
            "DATA_QUAL",
            "LAT_CONFIG",
        }

        missing = required_columns - columns

        if missing:
            raise RuntimeError(
                f"{path.name}: missing columns {sorted(missing)}"
            )

        # Force Astropy to read the table, revealing truncated files.
        _ = len(hdul["SC_DATA"].data)


def merge_spacecraft_files(
    input_files: list[Path],
    output_file: Path,
) -> None:
    """Merge weekly SC_DATA tables into one spacecraft FITS file."""

    if not input_files:
        raise RuntimeError("No weekly spacecraft files were found.")

    print(f"Found {len(input_files)} weekly spacecraft files.")

    tables = []

    first_primary_header = None
    first_sc_header = None
    last_sc_header = None

    for index, path in enumerate(input_files, start=1):
        print(
            f"[{index:4d}/{len(input_files):4d}] "
            f"Reading {path.name}"
        )

        validate_weekly_file(path)

        with fits.open(path, memmap=True) as hdul:
            if first_primary_header is None:
                first_primary_header = hdul[0].header.copy()
                first_sc_header = hdul["SC_DATA"].header.copy()

            last_sc_header = hdul["SC_DATA"].header.copy()

            table = Table(hdul["SC_DATA"].data)
            tables.append(table)

    print("Concatenating SC_DATA tables...")

    merged_table = vstack(
        tables,
        join_type="exact",
        metadata_conflicts="silent",
    )

    # Sort explicitly by start time, even if filenames were already sorted.
    merged_table.sort("START")

    start = np.asarray(merged_table["START"])
    stop = np.asarray(merged_table["STOP"])

    if np.any(stop <= start):
        bad_count = int(np.sum(stop <= start))
        raise RuntimeError(
            f"Found {bad_count} rows with STOP <= START."
        )

    # Create a fresh binary table from the combined rows.
    sc_hdu = fits.BinTableHDU(
        data=merged_table.as_array(),
        name="SC_DATA",
    )

    # Restore useful metadata from the first weekly file.
    structural_prefixes = (
        "TTYPE",
        "TFORM",
        "TUNIT",
        "TDIM",
        "TNULL",
        "TSCAL",
        "TZERO",
        "TDISP",
    )

    structural_keywords = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "CHECKSUM",
        "DATASUM",
    }

    for card in first_sc_header.cards:
        keyword = card.keyword

        if keyword in structural_keywords:
            continue

        if keyword.startswith(structural_prefixes):
            continue

        try:
            sc_hdu.header[keyword] = (
                card.value,
                card.comment,
            )
        except (ValueError, TypeError):
            pass

    sc_hdu.header["EXTNAME"] = "SC_DATA"

    # Equivalent in spirit to:
    # lastkey='TSTOP,DATE-END'
    for keyword in ("TSTOP", "DATE-END"):
        if keyword in last_sc_header:
            sc_hdu.header[keyword] = last_sc_header[keyword]

    # Make time coverage internally consistent.
    sc_hdu.header["TSTART"] = float(start.min())
    sc_hdu.header["TSTOP"] = float(stop.max())

    primary_hdu = fits.PrimaryHDU(
        header=first_primary_header,
    )

    # Update relevant primary-header end-time keywords too.
    for keyword in ("TSTOP", "DATE-END"):
        if keyword in last_sc_header:
            primary_hdu.header[keyword] = last_sc_header[keyword]

    primary_hdu.header["TSTART"] = float(start.min())
    primary_hdu.header["TSTOP"] = float(stop.max())

    hdul_out = fits.HDUList([
        primary_hdu,
        sc_hdu,
    ])

    print(f"Writing {output_file}")

    hdul_out.writeto(
        output_file,
        overwrite=True,
        checksum=True,
    )


def validate_merged_file(path: Path) -> None:
    """Validate the completed merged spacecraft file."""

    print("\nValidating merged file...")

    with fits.open(
        path,
        checksum=True,
        memmap=True,
    ) as hdul:
        hdul.info()

        extensions = [hdu.name for hdu in hdul]

        if "SC_DATA" not in extensions:
            raise RuntimeError("Merged SC_DATA extension is missing.")

        data = hdul["SC_DATA"].data

        start = np.asarray(data["START"])
        stop = np.asarray(data["STOP"])

        print(f"\nRows       : {len(data):,}")
        print(f"First START: {start.min()}")
        print(f"Last STOP  : {stop.max()}")
        print(
            "START sorted:",
            bool(np.all(np.diff(start) >= 0)),
        )
        print(
            "Invalid rows:",
            int(np.sum(stop <= start)),
        )
        print(
            "Overlapping adjacent intervals:",
            int(np.sum(start[1:] < stop[:-1])),
        )

        print(
            "IN_SAA present:",
            "IN_SAA" in hdul["SC_DATA"].columns.names,
        )


def main() -> None:
    input_files = sorted(
        SPACECRAFT_DIR.glob(FILE_PATTERN)
    )

    # Ensure the output is never included as an input.
    input_files = [
        path
        for path in input_files
        if path.resolve() != OUTPUT_FILE.resolve()
    ]

    try:
        merge_spacecraft_files(
            input_files=input_files,
            output_file=OUTPUT_FILE,
        )

        validate_merged_file(OUTPUT_FILE)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print("\nMerge completed successfully.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
