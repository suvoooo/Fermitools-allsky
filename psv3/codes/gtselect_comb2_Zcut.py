'''
this file prepares for the zenit angle cut on the photon list combined files
from DR3 release; the cuts are as follows

50 MeV - 100 MeV  : ZMax = 80 deg
100 MeV - 300 MeV : ZMax = 90 deg
300 MeV - 1 GeV   : ZMax = 100 deg
> 1 GeV           : ZMax = 105 deg
'''

from pathlib import Path
import sys

import gt_apps


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MASTER_FILE = Path("./lat_alldata_16yrs.fits")
OUTPUT_DIR = Path("./lat_all_data_Zcuts")

EVCLASS = 128
EVTYPE = 3 # select front and back both, no more separate treatement

RA = 0.0
DEC = 0.0
RADIUS = 180.0

TMIN = "INDEF"
TMAX = "INDEF"


ENERGY_SELECTIONS = [
    {
        "label": "30M_100M",
        "emin": 30.0,
        "emax": 100.0,
        "zmax": 80.0,
    },
    {
        "label": "100M_300M",
        "emin": 100.0,
        "emax": 300.0,
        "zmax": 90.0,
    },
    {
        "label": "300M_1G",
        "emin": 300.0,
        "emax": 1_000.0,
        "zmax": 100.0,
    },
    {
        "label": "1G_1T",
        "emin": 1_000.0,
        "emax": 1_000_000.0,
        "zmax": 105.0,
    },
]


def run_gtselect(
    input_file: Path,
    output_file: Path,
    emin: float,
    emax: float,
    zmax: float,
) -> None:
    """Run one energy-dependent gtselect operation."""

    print(
        f"\nRunning gtselect:\n"
        f"  input  : {input_file}\n"
        f"  output : {output_file}\n"
        f"  energy : {emin:g}--{emax:g} MeV\n"
        f"  zmax   : {zmax:g} deg"
    )

    gt_apps.filter["infile"] = str(input_file.resolve())
    gt_apps.filter["outfile"] = str(output_file.resolve())

    gt_apps.filter["ra"] = RA
    gt_apps.filter["dec"] = DEC
    gt_apps.filter["rad"] = RADIUS

    gt_apps.filter["tmin"] = TMIN
    gt_apps.filter["tmax"] = TMAX

    gt_apps.filter["emin"] = emin
    gt_apps.filter["emax"] = emax
    gt_apps.filter["zmax"] = zmax

    gt_apps.filter["evclass"] = EVCLASS
    gt_apps.filter["evtype"] = EVTYPE

    gt_apps.filter["clobber"] = "yes"

    gt_apps.filter.run()


def main() -> None:
    if not MASTER_FILE.exists():
        print(f"ERROR: Master event file not found:\n{MASTER_FILE}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created_files = []

    for selection in ENERGY_SELECTIONS:
        output_file = OUTPUT_DIR / (
            f"lat_selected_"
            f"{selection['label']}_"
            f"z{selection['zmax']:g}.fits"
        )

        run_gtselect(
            input_file=MASTER_FILE,
            output_file=output_file,
            emin=selection["emin"],
            emax=selection["emax"],
            zmax=selection["zmax"],
        )

        created_files.append(output_file)

    print("\nFinished successfully.")
    print("Created files:")

    for filename in created_files:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
