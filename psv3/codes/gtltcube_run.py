from pathlib import Path
import sys

import gt_apps


# ------------------------------------------------------------------
# file paths: gti and spacecraft
# ------------------------------------------------------------------

BASE_DIR = Path(
    "/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs"
)

EVENT_DIR = (
    BASE_DIR
    / "photon"
    / "lat_all_data_Zcuts_gti"
)

SPACECRAFT_FILE = (
    BASE_DIR
    / "spacecraft"
    / "lat_spacecraft_weekly_merged_astropy.fits"
)

OUTPUT_DIR = (
    BASE_DIR
    / "photon"
    / "livetime_cubes"
)


# ------------------------------------------------------------------
# Livetime-cube jobs
# ------------------------------------------------------------------

LTCUBE_JOBS = [
    {
        "event_file": "lat_selected_30M_100M_z80_gti.fits",
        "output_file": "lat_selected_30M_100M_z80_ltcube.fits",
        "zmax": 80.0,
    },
    {
        "event_file": "lat_selected_100M_300M_z90_gti.fits",
        "output_file": "lat_selected_100M_300M_z90_ltcube.fits",
        "zmax": 90.0,
    },
    {
        "event_file": "lat_selected_300M_1G_z100_gti.fits",
        "output_file": "lat_selected_300M_1G_z100_ltcube.fits",
        "zmax": 100.0,
    },
    {
        "event_file": "lat_selected_1G_1T_z105_gti.fits",
        "output_file": "lat_selected_1G_1T_z105_ltcube.fits",
        "zmax": 105.0,
    },
]


# standard gtltcube angular binning
DCOSTHETA = 0.025
BINSZ = 1.0
PHIBINS = 0


def run_gtltcube(
    event_file: Path,
    output_file: Path,
    zmax: float,
) -> None:
    """Generate one livetime cube."""

    print("\nRunning gtltcube")
    print(f"  event file : {event_file}")
    print(f"  spacecraft : {SPACECRAFT_FILE}")
    print(f"  output     : {output_file}")
    print(f"  zmax       : {zmax:g} deg")
    print(f"  dcostheta  : {DCOSTHETA}")
    print(f"  binsz      : {BINSZ} deg")
    print(f"  phibins    : {PHIBINS}")

    gt_apps.expCube["evfile"] = str(event_file.resolve())
    gt_apps.expCube["scfile"] = str(SPACECRAFT_FILE.resolve())
    gt_apps.expCube["outfile"] = str(output_file.resolve())

    gt_apps.expCube["zmax"] = zmax
    gt_apps.expCube["dcostheta"] = DCOSTHETA
    gt_apps.expCube["binsz"] = BINSZ
    gt_apps.expCube["phibins"] = PHIBINS

    gt_apps.expCube["clobber"] = "yes"
    gt_apps.expCube["chatter"] = 2

    gt_apps.expCube.run()


def main() -> None:
    if not EVENT_DIR.exists():
        print(f"ERROR: Event directory not found:\n{EVENT_DIR}")
        sys.exit(1)

    if not SPACECRAFT_FILE.exists():
        print(f"ERROR: Spacecraft file not found:\n{SPACECRAFT_FILE}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_files = []

    for job in LTCUBE_JOBS:
        path = EVENT_DIR / job["event_file"]

        if not path.exists():
            missing_files.append(path)

    if missing_files:
        print("ERROR: Missing GTI event files:")

        for path in missing_files:
            print(f"  {path}")

        sys.exit(1)

    created_files = []

    for job in LTCUBE_JOBS:
        event_file = EVENT_DIR / job["event_file"]
        output_file = OUTPUT_DIR / job["output_file"]

        run_gtltcube(
            event_file=event_file,
            output_file=output_file,
            zmax=job["zmax"],
        )

        created_files.append(output_file)

    print("\nAll livetime cubes created successfully:")

    for path in created_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
