#!/usr/bin/env python3
"""
Choose the reference tile for calibration: the first tile that is not flagged in the metafits.

Usage:
    python find_reference_tile.py 1476960704.metafits

The tile number is printed on one line, so a shell script can capture it:
    refant=$(python find_reference_tile.py "${metafits}")

Why a reference tile is needed
------------------------------
Calibration only fixes the phase DIFFERENCES between tiles. Adding the same phase to every
tile leaves the visibilities unchanged, so the overall phase is arbitrary. To make solutions
comparable, every tile's phase is measured relative to one tile, the reference tile.
A flagged tile has no data, so it cannot be the reference.

What this script can and cannot know
------------------------------------
It only sees the flags recorded in the metafits (tiles known to be bad when the observation
was taken). A tile that is unflagged here can still turn out to have bad data. Look at the
calibration plots to check that the reference tile behaves well.
"""

import argparse
import sys

from astropy.io import fits


def read_tile_inputs(metafits_file):
    """
    Read the tile table (called TILEDATA) from the metafits file.

    The table has one row for each input of a tile. Every tile has two inputs, one for
    the X polarisation and one for the Y polarisation, so every tile appears twice.

    Returns two lists of the same length:
        antenna_numbers: the tile number of each row (0, 0, 1, 1, 2, 2, ...)
        flags:           the flag of each row (0 means not flagged, anything else means flagged)
    """
    with fits.open(metafits_file) as hdu_list:
        tile_table = hdu_list["TILEDATA"].data
        antenna_numbers = list(tile_table["Antenna"])
        flags = list(tile_table["Flag"])
    return antenna_numbers, flags


def find_first_unflagged_tile(antenna_numbers, flags):
    """
    Return the number of the first tile whose X and Y inputs are both unflagged.
    Return None if every tile is flagged.
    """
    # Step 1: for each tile, find out whether any of its two inputs is flagged.
    tile_is_flagged = {}
    for antenna_number, flag in zip(antenna_numbers, flags):
        antenna_number = int(antenna_number)
        if antenna_number not in tile_is_flagged:
            tile_is_flagged[antenna_number] = False
        if int(flag) != 0:
            tile_is_flagged[antenna_number] = True

    # Step 2: go through the tiles in order and return the first one that is not flagged.
    for antenna_number in sorted(tile_is_flagged):
        if not tile_is_flagged[antenna_number]:
            return antenna_number

    return None


def count_flagged_tiles(antenna_numbers, flags):
    """Return how many tiles have at least one flagged input, and how many tiles there are."""
    tile_is_flagged = {}
    for antenna_number, flag in zip(antenna_numbers, flags):
        antenna_number = int(antenna_number)
        if antenna_number not in tile_is_flagged:
            tile_is_flagged[antenna_number] = False
        if int(flag) != 0:
            tile_is_flagged[antenna_number] = True

    number_flagged = 0
    for antenna_number in tile_is_flagged:
        if tile_is_flagged[antenna_number]:
            number_flagged = number_flagged + 1
    return number_flagged, len(tile_is_flagged)


def main():
    parser = argparse.ArgumentParser(
        description="Print the number of the first tile that is not flagged in a metafits file."
    )
    parser.add_argument("metafits", help="the metafits file of the observation")
    arguments = parser.parse_args()

    # Reading can fail if the file is missing, is not a FITS file, or has no TILEDATA table.
    try:
        antenna_numbers, flags = read_tile_inputs(arguments.metafits)
    except (OSError, KeyError) as error:
        print(f"ERROR: cannot read the tile table from {arguments.metafits}: {error}", file=sys.stderr)
        sys.exit(1)

    reference_tile = find_first_unflagged_tile(antenna_numbers, flags)
    if reference_tile is None:
        print(f"ERROR: every tile is flagged in {arguments.metafits}.", file=sys.stderr)
        sys.exit(1)

    number_flagged, number_of_tiles = count_flagged_tiles(antenna_numbers, flags)

    # Messages for people go to stderr (they end up in the .e log).
    # Only the result goes to stdout, so that "refant=$(...)" in a shell script captures just the number.
    print(f"INFO: {number_flagged} of {number_of_tiles} tiles are flagged in the metafits.", file=sys.stderr)
    print(f"INFO: the first unflagged tile is number {reference_tile}.", file=sys.stderr)
    print(reference_tile)


if __name__ == "__main__":
    main()