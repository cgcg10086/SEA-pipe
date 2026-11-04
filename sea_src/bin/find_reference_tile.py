#!/usr/bin/env python3
"""
Choose the reference tile for calibration: the first tile that is not flagged in the metafits.

Usage:
    python find_reference_tile.py 1476960704.metafits

Tiles that you flagged by hand (they are not in the metafits) can be skipped as well:
    python find_reference_tile.py 1476960704.metafits --skip-file 1476960704_flagged_tiles.txt

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
import os
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


def read_skipped_tiles(skip_file):
    """
    Read the tile numbers that were flagged by hand, from a text file.
    The file holds whole numbers separated by spaces or line breaks.

    Return an empty list if the file does not exist (no tile has been flagged by hand yet).
    A word that is not a whole number makes int() raise a ValueError, which main() reports.
    """
    skipped_tiles = []
    if not os.path.exists(skip_file):
        return skipped_tiles
    with open(skip_file) as file_handle:
        for word in file_handle.read().split():
            skipped_tiles.append(int(word))
    return skipped_tiles


def make_flag_table(antenna_numbers, flags, skipped_tiles):
    """
    Return a dictionary that says, for every tile number, whether the tile is flagged.
    A tile is flagged if any of its two inputs is flagged in the metafits, or if it is in skipped_tiles.
    """
    tile_is_flagged = {}
    for antenna_number, flag in zip(antenna_numbers, flags):
        antenna_number = int(antenna_number)
        if antenna_number not in tile_is_flagged:
            tile_is_flagged[antenna_number] = False
        if int(flag) != 0:
            tile_is_flagged[antenna_number] = True

    # Tiles that were flagged by hand count as flagged too.
    for antenna_number in skipped_tiles:
        tile_is_flagged[antenna_number] = True

    return tile_is_flagged


def find_first_unflagged_tile(tile_is_flagged):
    """Return the number of the first tile that is not flagged, or None if every tile is flagged."""
    for antenna_number in sorted(tile_is_flagged):
        if not tile_is_flagged[antenna_number]:
            return antenna_number
    return None


def count_flagged_tiles(tile_is_flagged):
    """Return how many tiles are flagged, and how many tiles there are."""
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
    parser.add_argument(
        "--skip-file",
        default=None,
        help="text file with the numbers of tiles that were flagged by hand; these tiles are never chosen",
    )
    arguments = parser.parse_args()

    # Reading can fail if a file is missing, is not a FITS file, has no TILEDATA table,
    # or if the skip file holds something that is not a whole number.
    skipped_tiles = []
    try:
        antenna_numbers, flags = read_tile_inputs(arguments.metafits)
        if arguments.skip_file is not None:
            skipped_tiles = read_skipped_tiles(arguments.skip_file)
    except (OSError, KeyError, ValueError) as error:
        print(f"ERROR: cannot read the tile information: {error}", file=sys.stderr)
        sys.exit(1)

    tile_is_flagged = make_flag_table(antenna_numbers, flags, skipped_tiles)

    reference_tile = find_first_unflagged_tile(tile_is_flagged)
    if reference_tile is None:
        print(f"ERROR: every tile is flagged (in {arguments.metafits} or by hand).", file=sys.stderr)
        sys.exit(1)

    number_flagged, number_of_tiles = count_flagged_tiles(tile_is_flagged)

    # Messages for people go to stderr (they end up in the .e log).
    # Only the result goes to stdout, so that "refant=$(...)" in a shell script captures just the number.
    print(f"INFO: {number_flagged} of {number_of_tiles} tiles are flagged (metafits and by hand).", file=sys.stderr)
    if len(skipped_tiles) > 0:
        print(f"INFO: tiles flagged by hand and skipped: {skipped_tiles}", file=sys.stderr)
    print(f"INFO: the first unflagged tile is number {reference_tile}.", file=sys.stderr)
    print(reference_tile)


if __name__ == "__main__":
    main()