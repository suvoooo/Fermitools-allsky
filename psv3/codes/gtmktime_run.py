'''
after the second gtselect run i.e. making sure proper zmax cuts
now we select good time intervals within those fits files
'''

from pathlib import Path
import sys

import gt_apps


BASE_DIR = Path("/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs")

GTSELECT_DIR = (
    BASE_DIR
    / "photon"
    / "lat_all_data_Zcuts"
)

SPACECRAFT_FILE = (
    BASE_DIR
    / "spacecraft"
    / "lat_spacecraft_weekly_merged_astropy.fits"
)

OUTPUT_DIR = (
    BASE_DIR
    / "photon"
    / "lat_all_data_Zcuts_gti"
)


GTSELECT_FILES = [
    "lat_selected_30M_100M_z80.fits",
    "lat_selected_100M_300M_z90.fits",
    "lat_selected_300M_1G_z100.fits",
    "lat_selected_1G_1T_z105.fits",
]


GTI_FILTER = (
    "(DATA_QUAL>0)"
    "&&(LAT_CONFIG==1)"
    "&&(IN_SAA!=T)"
)


def run_gtmktime(
    input_file: Path,
    output_file: Path,
) -> None:
    """Apply GTI cuts to one selected LAT event file."""

    print("\nRunning gtmktime")
    print(f"  input      : {input_file}")
    print(f"  spacecraft : {SPACECRAFT_FILE}")
    print(f"  output     : {output_file}")
    print(f"  filter     : {GTI_FILTER}")

    gt_apps.maketime["scfile"] = str(
        SPACECRAFT_FILE.resolve()
    )

    gt_apps.maketime["evfile"] = str(
        input_file.resolve()
    )

    gt_apps.maketime["outfile"] = str(
        output_file.resolve()
    )

    gt_apps.maketime["filter"] = GTI_FILTER

    # Appropriate for a full-sky event file.
    gt_apps.maketime["roicut"] = "no"

    gt_apps.maketime["clobber"] = "yes"

    gt_apps.maketime.run()


def main() -> None:
    if not GTSELECT_DIR.exists():
        print(
            "ERROR: gtselect directory not found:\n"
            f"{GTSELECT_DIR}"
        )
        sys.exit(1)

    if not SPACECRAFT_FILE.exists():
        print(
            "ERROR: spacecraft file not found:\n"
            f"{SPACECRAFT_FILE}"
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_files = [
        GTSELECT_DIR / filename
        for filename in GTSELECT_FILES
    ]

    missing_files = [
        path
        for path in input_files
        if not path.exists()
    ]

    if missing_files:
        print("ERROR: Missing event files:")

        for path in missing_files:
            print(f"  {path}")

        sys.exit(1)

    created_files = []

    for input_file in input_files:
        output_file = (
            OUTPUT_DIR
            / f"{input_file.stem}_gti.fits"
        )

        run_gtmktime(
            input_file=input_file,
            output_file=output_file,
        )

        created_files.append(output_file)

    print("\nFinished successfully.")
    print("Created files:")

    for path in created_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
