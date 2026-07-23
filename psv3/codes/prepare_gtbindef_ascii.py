from pathlib import Path

import numpy as np


BASE_DIR = Path(
    "/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs"
)

OUTPUT_DIR = (
    BASE_DIR
    / "photon"
    / "energy_bin_definitions"
)

BINS_PER_DECADE = 6

GLOBAL_EMIN = 30.0
GLOBAL_EMAX = 1_000_000.0  # MeV = 1 TeV


ENERGY_RANGES = [
    {
        "label": "30M_100M",
        "emin": 30.0,
        "emax": 100.0,
    },
    {
        "label": "100M_300M",
        "emin": 100.0,
        "emax": 300.0,
    },
    {
        "label": "300M_1G",
        "emin": 300.0,
        "emax": 1_000.0,
    },
    {
        "label": "1G_1T",
        "emin": 1_000.0,
        "emax": 1_000_000.0,
    },
]


def make_global_edges(
    emin: float,
    emax: float,
    bins_per_decade: int,
) -> np.ndarray:
    """
    create log grid with approx. the requested number
    of bins per decade, then force the analysis boundaries into it.
    """

    log_step = 1.0 / bins_per_decade

    number_of_full_steps = int(
        np.floor(
            bins_per_decade
            * np.log10(emax / emin)
        )
    )

    edges = emin * 10.0 ** (
        np.arange(number_of_full_steps + 1)
        * log_step
    )

    # append the exact upper limit because 1 TeV is not exactly reached
    # by an integer number of 1/6-decade steps starting from 30 MeV.
    if not np.isclose(edges[-1], emax):
        edges = np.append(edges, emax)

    # Force boundaries corresponding to the separate zenith selections.
    required_edges = np.array(
        [
            30.0,
            100.0,
            300.0,
            1_000.0,
            1_000_000.0,
        ]
    )

    edges = np.concatenate([edges, required_edges])

    # round before unique to suppress tiny floating-point differences.
    edges = np.unique(np.round(edges, decimals=8))
    edges.sort()

    return edges


def select_edges_for_range(
    global_edges: np.ndarray,
    emin: float,
    emax: float,
) -> np.ndarray:
    """extract the bin edges belonging to one zenith-cut range."""

    mask = (
        (global_edges >= emin)
        & (global_edges <= emax)
    )

    edges = global_edges[mask]

    if not np.isclose(edges[0], emin):
        edges = np.insert(edges, 0, emin)

    if not np.isclose(edges[-1], emax):
        edges = np.append(edges, emax)

    return edges


def write_gtbindef_ascii(
    output_file: Path,
    edges: np.ndarray,
) -> None:
    """
    write the 2-col. ASCII format expected by gtbindef.

    Eeach row contains:
        lower_energy upper_energy

    energies are in MeV. as expected in gtbin....
    """

    with output_file.open("w", encoding="ascii") as handle:
        for lower, upper in zip(edges[:-1], edges[1:]):
            handle.write(
                f"{lower:.8f} {upper:.8f}\n"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_edges = make_global_edges(
        emin=GLOBAL_EMIN,
        emax=GLOBAL_EMAX,
        bins_per_decade=BINS_PER_DECADE,
    )

    print("Complete energy grid")
    print("--------------------")

    for index, edge in enumerate(global_edges):
        print(f"{index:02d}: {edge:14.6f} MeV")

    print(
        f"\nTotal number of final bins: "
        f"{len(global_edges) - 1}"
    )

    for energy_range in ENERGY_RANGES:
        edges = select_edges_for_range(
            global_edges=global_edges,
            emin=energy_range["emin"],
            emax=energy_range["emax"],
        )

        output_file = (
            OUTPUT_DIR
            / f"energy_bins_{energy_range['label']}.txt"
        )

        write_gtbindef_ascii(
            output_file=output_file,
            edges=edges,
        )

        print(
            f"\n{energy_range['label']}: "
            f"{len(edges) - 1} bins"
        )
        print(f"Written: {output_file}")

        for lower, upper in zip(edges[:-1], edges[1:]):
            print(
                f"  {lower:12.4f} -- "
                f"{upper:12.4f} MeV"
            )


if __name__ == "__main__":
    main()
