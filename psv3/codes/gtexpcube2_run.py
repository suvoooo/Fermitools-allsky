from pathlib import Path
import sys

from GtApp import GtApp


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

BASE_DIR = Path(
    "/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs"
)

# path to files.... gtbin, livetime and exposure cubes (output from this)...
GTBIN_DIR = BASE_DIR
LTCUBE_DIR = BASE_DIR / "photon" / "livetime_cubes"
OUTPUT_DIR = BASE_DIR / "photon" / "exposure_cubes"


# ------------------------------------------------------------------
# Exposure jobs
# ------------------------------------------------------------------

EXPOSURE_JOBS = [
    {
        "label": "30M_100M_z80",
        "ltcube": "lat_selected_30M_100M_z80_ltcube.fits",
        "gtbin": "lat_selected_30M_100M_z80_gtbin.fits",
        "output": "lat_selected_30M_100M_z80_expcube.fits",
    },
    {
        "label": "100M_300M_z90",
        "ltcube": "lat_selected_100M_300M_z90_ltcube.fits",
        "gtbin": "lat_selected_100M_300M_z90_gtbin.fits",
        "output": "lat_selected_100M_300M_z90_expcube.fits",
    },
    {
        "label": "300M_1G_z100",
        "ltcube": "lat_selected_300M_1G_z100_ltcube.fits",
        "gtbin": "lat_selected_300M_1G_z100_gtbin.fits",
        "output": "lat_selected_300M_1G_z100_expcube.fits",
    },
    {
        "label": "1G_1T_z105",
        "ltcube": "lat_selected_1G_1T_z105_ltcube.fits",
        "gtbin": "lat_selected_1G_1T_z105_gtbin.fits",
        "output": "lat_selected_1G_1T_z105_expcube.fits",
    },
]


IRFS = "P8R3_SOURCE_V3"
EVTYPE = 3


def run_gtexpcube2(
    ltcube_file: Path,
    counts_cube_file: Path,
    output_file: Path,
) -> None:
    """Generate one HEALPix binned-exposure cube."""

    print("\nRunning gtexpcube2")
    print(f"  livetime cube : {ltcube_file}")
    print(f"  counts cube   : {counts_cube_file}")
    print(f"  output        : {output_file}")
    print(f"  IRFs          : {IRFS}")
    print(f"  event type    : {EVTYPE}")
    print("  energy layers : CENTER")

    exp_cube_2 = GtApp(
        "gtexpcube2",
        "Likelihood",
    )

    exp_cube_2["infile"] = str(ltcube_file.resolve())
    exp_cube_2["cmap"] = str(counts_cube_file.resolve())
    exp_cube_2["outfile"] = str(output_file.resolve())

    exp_cube_2["irfs"] = "P8R3_SOURCE_V3"
    exp_cube_2["evtype"] = 3

    exp_cube_2["coordsys"] = "GAL"
    exp_cube_2["hpx_ordering_scheme"] = "RING"

    # Choose 11 for exact agreement with the count cube,
    # or 9/10 for a smaller, smoother exposure representation.
    exp_cube_2["hpx_order"] = 11

    exp_cube_2["bincalc"] = "EDGE"
    exp_cube_2["ignorephi"] = "no"

    exp_cube_2["thmin"] = 0.0
    exp_cube_2["thmax"] = 180.0

    exp_cube_2["clobber"] = "yes"

    exp_cube_2.run()


def main() -> None:
    if not GTBIN_DIR.exists():
        print(f"ERROR: gtbin directory missing:\n{GTBIN_DIR}")
        sys.exit(1)

    if not LTCUBE_DIR.exists():
        print(f"ERROR: livetime-cube directory missing:\n{LTCUBE_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for job in EXPOSURE_JOBS:
        ltcube_file = LTCUBE_DIR / job["ltcube"]
        counts_cube_file = GTBIN_DIR / job["gtbin"]
        output_file = OUTPUT_DIR / job["output"]

        if not ltcube_file.exists():
            print(
                "ERROR: Livetime cube not found:\n"
                f"{ltcube_file}"
            )
            sys.exit(1)

        if not counts_cube_file.exists():
            print(
                "ERROR: Count cube not found:\n"
                f"{counts_cube_file}"
            )
            sys.exit(1)

        run_gtexpcube2(
            ltcube_file=ltcube_file,
            counts_cube_file=counts_cube_file,
            output_file=output_file,
        )

    print("\nAll exposure cubes completed successfully.")

    for job in EXPOSURE_JOBS:
        print(f"  {OUTPUT_DIR / job['output']}")


if __name__ == "__main__":
    main()
