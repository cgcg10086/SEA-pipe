#!/usr/bin/env python3
"""
Where is every observation of a list in the pipeline, and what has to be done next?

It looks at the files that each step leaves behind (not at the SLURM logs), so it can be run at any time, on 10 or 1000 observations.
It only reads. It never changes your data. It needs nothing but Python 3.6 or newer, so it runs on the login node without the container.

usage:
    python3 pipeline_summary.py <project_dir> <obs_list> [--tiles <tile_list>] [--max-tile N]

    <tile_list> is your manual flag list: one line per observation, the obsid and then the tiles to flag, separated by spaces.
    An observation that you looked at and that needs no extra tile:    1444763232   none
    An observation that you have not decided on yet:                   1444763232   # TODO
    Anything after a # is a comment: write there why you chose those tiles.
    --max-tile N   the largest tile number of your array; a larger number in the tile list is reported as a probable typo.

The pipeline runs autocal twice:
    pass 1   before you look at the solutions
             (you look, write the tile list, and obs_autoflag.sh flags the tiles and writes <obsid>_flagged_tiles.txt)
    pass 2   after the flags. A solution is only up to date if it is NEWER than <obsid>_flagged_tiles.txt.

It writes the lists of the observations into the summary folder, one list per STEP THAT THE OBSERVATIONS HAVE JUST COMPLETED.
Every observation is in exactly one of them, so the lists add up to the whole observation list, and the next step is the one after it.
(obs_list stands for the name of your list without .txt; a list that would be empty is not written.)

    obs_list_not_downloaded.txt   waiting for the download
    obs_list_downloaded.txt       downloaded; autocal has not been run yet
    obs_list_autocal_failed.txt   autocal ended in an unexpected way: run it again
    obs_list_autocal1.txt         first autocal done; waiting for your decision about the tiles
    obs_list_inspected.txt        your decision is in the tile list; the flags are not applied yet
    obs_list_flagged.txt          the flags are applied; the second autocal is next
    obs_list_autocal2.txt         second autocal done; applying the calibration is next
    obs_list_qc_failed.txt        the quality check refused the solutions after the flags: a person decides
    obs_list_applied.txt          calibration applied; uvflag is next
    obs_list_uvflagged.txt        uvflag done; the image is next
    obs_list_imaged.txt           image done; the cube is next
    obs_list_cube.txt             everything is done

(they are in  <project_dir>/summary_<obs_list without .txt>/ , together with these files:)
    table.csv           one line per observation, one column per step, its state, and what to do next
    warnings.txt        things that do not block but are worth a look
    summary.txt         the counts (also printed)
and one line is added to <project_dir>/summary_history.log

The actions, in the order they are checked:
    download            no folder, or no measurement set
    autocal             first run missing, or a run ended in an unexpected way, or the solutions are older than the flags (pass 2 needed)
    fix_tile_list       the observation is in the tile list more than once
    inspect_and_list    pass 1 is done, but the observation is not in the tile list yet, or is marked TODO: somebody has to look at it
    flag                the tile list asks for something that obs_autoflag.sh has not recorded yet
    review_qc_failed    the quality check refused the solutions after the flags: expected, a person decides
    apply, uvflag, image, cube
    done
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


# the autocal outcomes that are NOT the quality check refusing the solutions: something went wrong that nobody expected
UNEXPECTED_AUTOCAL = ["no_solution", "task_did_not_finish", "task_failed_after_solution"]

# The step that an observation has just completed, in the order of the pipeline, and the step that comes next.
# The state is the name of the list that the observation goes into: <obs_list>_<state>.txt
STATES = [
    ("not_downloaded", "download"),
    ("downloaded", "autocal (first)"),
    ("autocal_failed", "autocal again"),
    ("autocal1", "inspect (you)"),
    ("inspected", "flag"),
    ("flagged", "autocal (second)"),
    ("autocal2", "apply"),
    ("qc_failed", "you decide"),
    ("applied", "uvflag"),
    ("uvflagged", "image"),
    ("imaged", "cube"),
    ("cube", "nothing, done"),
]

TABLE_COLUMNS = [
    "obsid", "download", "flags", "n_flagged_tiles", "tiles_missing_in_record", "autocal", "autocal_pass",
    "apply", "uvflag", "image", "cube", "n_frames", "state", "next", "autocal_status_line", "checked",
]


# ------------------------------------------------------------------------------------------------------------------
# Reading the input lists
# ------------------------------------------------------------------------------------------------------------------

def read_obs_list(path):
    """The obsids of the list, in order. Empty lines and lines that start with # are skipped."""
    obsids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line == "":
            continue
        if line.startswith("#"):
            continue
        obsids.append(line)
    return obsids


def whole_numbers(text):
    """All the whole numbers in a text, as a set. For example '3 9 12' gives {3, 9, 12}."""
    return set(int(word) for word in re.findall(r"\b\d+\b", text))


def read_tile_list(path, max_tile=None):
    """
    The tile list as a dictionary:  obsid -> list of the lines of that obsid.
    Each line is a dictionary:   "tiles" = the set of tiles asked for,   "undecided" = True if the comment of the line says TODO.
    More than one line for the same obsid is a mistake that the summary reports.
    The second thing returned is a list of the problems found while reading: a word that is not a whole number,
    a tile written twice in one line, a tile number above max_tile.
    """
    lines_of = {}
    problems = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        words = line.split()
        if len(words) == 0:
            continue
        obsid = words[0]
        if not obsid.isdigit():
            continue                                    # a comment line, or something else that is not an obsid

        text, hash_sign, comment = line.partition("#")  # text is everything in front of a comment
        tiles = set()
        for word in text.split()[1:]:
            if word.lower() == "none":
                continue                                # "none" says that you looked and that there is nothing to flag
            if not word.isdigit():
                problems.append(f"line {number} of the tile list ({obsid}): '{word}' is not a whole number; separate the tiles with spaces")
                continue
            tile = int(word)
            if tile in tiles:
                problems.append(f"line {number} of the tile list ({obsid}): tile {tile} is written twice")
            if max_tile is not None and tile > max_tile:
                problems.append(f"line {number} of the tile list ({obsid}): tile {tile} is above the largest tile number {max_tile}")
            tiles.add(tile)

        undecided = "todo" in comment.lower()
        lines_of.setdefault(obsid, []).append({"tiles": tiles, "undecided": undecided})
    return lines_of, problems


def read_text_or_empty(path):
    if path.exists():
        return path.read_text()
    return ""


# ------------------------------------------------------------------------------------------------------------------
# One function per step. Each one looks at the files of one observation and returns what it found.
# ------------------------------------------------------------------------------------------------------------------

def check_download(folder, obsid):
    if not folder.is_dir():
        return "no_folder"
    if not (folder / (obsid + ".ms")).is_dir():
        return "MS_MISSING"
    return "ok"


def check_flags(obsid, record, tile_list):
    """
    What you asked for in the tile list, against what obs_autoflag.sh recorded in <obsid>_flagged_tiles.txt.
    That record is written (also empty, when there is nothing to flag) every time obs_autoflag.sh has dealt with the observation,
    so its date is the moment your decision was applied.
    Returns: the flag state, how many tiles are recorded as flagged, the tiles that are asked for but not recorded,
    and the tiles that are recorded but are not in the tile list (flags cannot be undone, so those stay flagged).
    """
    recorded = set()
    if record.exists():
        recorded = whole_numbers(record.read_text())

    if tile_list is None:
        return "not_checked", len(recorded), [], []

    lines = tile_list.get(obsid, [])
    if len(lines) > 1:
        return "DUPLICATE_IN_LIST", len(recorded), [], []
    if len(lines) == 0:
        return "not_in_list", len(recorded), [], []

    requested = lines[0]["tiles"]
    missing = sorted(requested - recorded)
    extra = sorted(recorded - requested)

    if lines[0]["undecided"]:
        return "undecided", len(recorded), missing, extra
    if not record.exists():
        return "not_applied", len(recorded), missing, extra
    if len(missing) > 0:
        return "incomplete", len(recorded), missing, extra
    if len(requested) == 0:
        return "none_needed", len(recorded), missing, extra
    return "ok", len(recorded), missing, extra


def check_autocal(prefix, status, record):
    """
    The latest autocal run, and whether it came before or after your flags.
    prefix is the start of the names of the solution files:  <folder>/<obsid>_local_gleam_model_solutions_initial
    status is the line that the task wrote about itself (empty for older runs).
    """
    raw = Path(prefix + ".bin")
    good = Path(prefix + "_ref.bin")
    refused = Path(prefix + "_ref_failed.bin")

    solution = None
    if refused.exists():
        autocal = "qc_failed"                        # the quality check said no: expected
        solution = refused
    elif good.exists():
        autocal = "ok"
        solution = good
        if "exit=running" in status:
            autocal = "task_did_not_finish"
        elif status != "" and "exit=0 " not in status:
            autocal = "task_failed_after_solution"
    elif status != "" or raw.exists():
        autocal = "no_solution"
    else:
        autocal = "not_run"

    # pass 1 = made before your flags were applied (or no flags applied yet), pass 2 = made after
    autocal_pass = "n/a"
    if solution is not None:
        autocal_pass = "1_before_flags"
        if record.exists():
            if record.stat().st_mtime > solution.stat().st_mtime:
                autocal_pass = "1_before_flags"
            else:
                autocal_pass = "2_after_flags"
    return autocal, autocal_pass


def number_of_frames(cube):
    """The number of frames of a cube, from the NAXIS3 card of its FITS header (the header is made of cards of 80 characters)."""
    with open(str(cube), "rb") as handle:
        header = handle.read(14400).decode("ascii", errors="replace")
    for start in range(0, len(header), 80):
        card = header[start:start + 80]
        if card.startswith("NAXIS3 "):
            value = card.split("=")[1].split("/")[0]
            return int(value)
    return ""


def check_later_steps(folder, obsid):
    """The later steps leave one product each."""
    apply = "not_applied"
    if (folder / (obsid + "_calibration_applied.txt")).exists():
        apply = "applied"

    uvflag = "not_done"
    if (folder / (obsid + "_uvflag_done.txt")).exists():
        uvflag = "done"

    image = "missing"
    if (folder / (obsid + "_deep-MFS-image.fits")).exists():
        image = "ok"

    cube = "missing"
    n_frames = ""
    cube_file = folder / (obsid + "_transient-cube.fits")
    if cube_file.exists():
        cube = "ok"
        n_frames = number_of_frames(cube_file)

    return apply, uvflag, image, cube, n_frames


# ------------------------------------------------------------------------------------------------------------------
# What to do next: the first thing in this order that is not fine.
# ------------------------------------------------------------------------------------------------------------------

def decide_next(row):
    if row["download"] != "ok":
        return "download"
    if row["autocal"] == "not_run" and row["flags"] in ("not_applied", "incomplete"):
        return "flag"                                 # you decided on these tiles before (an earlier run): flag first, then calibrate only once
    if row["autocal"] == "not_run":
        return "autocal"
    if row["autocal"] in UNEXPECTED_AUTOCAL:
        return "autocal"
    if row["flags"] == "DUPLICATE_IN_LIST":
        return "fix_tile_list"
    if row["flags"] in ("not_in_list", "undecided"):
        return "inspect_and_list"
    if row["flags"] in ("not_applied", "incomplete"):
        return "flag"
    if row["flags"] != "not_checked" and row["autocal_pass"] == "1_before_flags":
        return "autocal"                              # the flags were applied after this solution was made: pass 2 is needed
    if row["autocal"] == "qc_failed":
        return "review_qc_failed"
    if row["apply"] != "applied":
        return "apply"
    if row["uvflag"] != "done":
        return "uvflag"
    if row["image"] != "ok":
        return "image"
    if row["cube"] != "ok":
        return "cube"
    return "done"


def decide_state(row):
    """The step that the observation has just completed. It follows from what decide_next found: the step that comes next."""
    next_action = row["next"]

    if next_action == "download":
        return "not_downloaded"
    if next_action == "autocal":
        if row["autocal"] == "not_run":
            return "downloaded"
        if row["autocal"] in UNEXPECTED_AUTOCAL:
            return "autocal_failed"
        return "flagged"                              # the flags were applied after these solutions were made: the second autocal is next
    if next_action in ("fix_tile_list", "inspect_and_list"):
        return "autocal1"
    if next_action == "flag":
        return "inspected"
    if next_action == "review_qc_failed":
        return "qc_failed"
    if next_action == "apply":
        if row["flags"] == "not_checked":
            return "autocal1"                         # without a tile list the summary cannot tell the two autocal runs apart
        return "autocal2"
    if next_action == "uvflag":
        return "applied"
    if next_action == "image":
        return "uvflagged"
    if next_action == "cube":
        return "imaged"
    return "cube"


# ------------------------------------------------------------------------------------------------------------------
# One observation
# ------------------------------------------------------------------------------------------------------------------

def check_observation(project_dir, obsid, tile_list, checked_at):
    folder = project_dir / obsid
    record = folder / (obsid + "_flagged_tiles.txt")
    prefix = str(folder / (obsid + "_local_gleam_model_solutions_initial"))

    # the line that the autocal task wrote about itself (see autocal.tmpl); the node name is of no interest here
    status = read_text_or_empty(folder / (obsid + "_autocal_status.txt")).strip()
    status = re.sub(r"\s+node=\S+", "", status)

    row = {"obsid": obsid, "autocal_status_line": status, "checked": checked_at}
    row["download"] = check_download(folder, obsid)
    row["flags"], row["n_flagged_tiles"], missing, extra = check_flags(obsid, record, tile_list)
    row["tiles_missing_in_record"] = " ".join(str(tile) for tile in missing)
    row["tiles_not_in_list"] = " ".join(str(tile) for tile in extra)          # only used for the warnings, it is not a column of the table
    row["autocal"], row["autocal_pass"] = check_autocal(prefix, status, record)
    row["apply"], row["uvflag"], row["image"], row["cube"], row["n_frames"] = check_later_steps(folder, obsid)
    row["next"] = decide_next(row)
    row["state"] = decide_state(row)
    return row


# ------------------------------------------------------------------------------------------------------------------
# Writing the results
# ------------------------------------------------------------------------------------------------------------------

def write_lists(project_dir, out_dir, rows, list_stem):
    """
    One list per state, in the summary folder:  <list_stem>_<state>.txt  (for example obs_list_autocal1.txt).
    Only these files are touched. The list of a state that nobody is in any more is removed, so that no old list stays behind.
    """
    for state, next_step in STATES:
        for old_file in [out_dir / (list_stem + "_" + state + ".txt"), project_dir / (list_stem + "_" + state + ".txt")]:    # the second one: where an older version wrote them
            if old_file.exists():
                old_file.unlink()

    # lists made by older versions of this program, inside the summary folder
    for pattern in ["next_*.txt", "done.txt", "*_downloaded.txt", "*_not_downloaded.txt"]:
        for old_file in out_dir.glob(pattern):
            old_file.unlink()

    for row in rows:
        list_file = out_dir / (list_stem + "_" + row["state"] + ".txt")
        with open(str(list_file), "a") as handle:
            handle.write(row["obsid"] + "\n")


def find_warnings(rows, tile_list, tile_list_name, tile_problems):
    """Things that do not block, but that somebody should look at."""
    warnings = []

    # mistakes in the tile list itself
    for problem in tile_problems:
        warnings.append(problem)
    if tile_list is not None:
        in_the_obs_list = set(row["obsid"] for row in rows)
        for obsid in tile_list:
            if obsid not in in_the_obs_list:
                warnings.append(f"{obsid}: is in {tile_list_name} but not in the observation list (a typo?)")

    for row in rows:
        if row["flags"] == "DUPLICATE_IN_LIST":
            warnings.append(f"{row['obsid']}: appears {len(tile_list[row['obsid']])} times in {tile_list_name}")
        if row["flags"] == "incomplete":
            warnings.append(f"{row['obsid']}: tiles in the list but not in the record of flagged tiles: {row['tiles_missing_in_record']}")
        if row["tiles_not_in_list"] != "":
            warnings.append(f"{row['obsid']}: tiles {row['tiles_not_in_list']} are flagged in the measurement set but are not in {tile_list_name} (flags cannot be undone)")

    # the cubes should all have the same number of frames: say which ones do not
    counts = Counter(row["n_frames"] for row in rows if row["n_frames"] != "")
    if len(counts) > 0:
        most_common = counts.most_common(1)[0][0]
        for row in rows:
            if row["n_frames"] != "" and row["n_frames"] != most_common:
                warnings.append(f"{row['obsid']}: the cube has {row['n_frames']} frames, most cubes have {most_common}")
    return warnings


def stage_of(status):
    """'... stage=hyperdrive_calibrate ...' gives 'hyperdrive_calibrate'."""
    match = re.search(r"stage=(\w+)", status)
    if match:
        return match.group(1)
    return "unknown"


def make_summary_text(list_name, rows, warnings, out_dir, checked_at):
    lines = [f"pipeline summary of {list_name}  ({len(rows)} observations, checked {checked_at})"]

    for column in ["download", "flags", "autocal", "autocal_pass", "apply", "uvflag", "image", "cube"]:
        counts = Counter(row[column] for row in rows)
        text = "   ".join(f"{number} {value}" for value, number in counts.most_common())
        lines.append(f"{column:<13} {text}")

    lines.append("the step that the observations have just completed (and the step that comes next):")
    counts = Counter(row["state"] for row in rows)
    for state, next_step in STATES:
        lines.append(f"   {counts[state]:5d}  {state:<15} next: {next_step}")

    unexpected = Counter((row["autocal"], stage_of(row["autocal_status_line"])) for row in rows if row["autocal"] in UNEXPECTED_AUTOCAL)
    if len(unexpected) > 0:
        lines.append("unexpected autocal failures (not the quality check), by where they stopped:")
        for (outcome, stage), number in unexpected.most_common():
            lines.append(f"   {number:5d}  {outcome} stage={stage}")

    if len(warnings) > 0:
        lines.append(f"warnings: {len(warnings)} (see {out_dir / 'warnings.txt'})")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------
# The program
# ------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Where is every observation of a list in the pipeline, and what has to be done next?")
    parser.add_argument("project_dir", help="the project folder, with one folder per observation")
    parser.add_argument("obs_list", help="the list of obsids, one per line")
    parser.add_argument("--tiles", help="your manual flag list: obsid, then the tiles to flag, one line per observation")
    parser.add_argument("--max-tile", type=int, help="the largest tile number of your array (the numbers in the titles of the solution plots); a larger number in the tile list is reported")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    if not project_dir.is_dir():
        # a bare project name, for example test6_tr0_nights: look for it in your scratch folder (GXSCRATCH, set by the profile)
        scratch = os.environ.get("GXSCRATCH")
        if scratch is not None:
            if (Path(scratch) / project_dir).is_dir():
                project_dir = Path(scratch) / project_dir
    if not project_dir.is_dir():
        sys.exit(f"I cannot find the project folder {args.project_dir} (also looked for it inside GXSCRATCH)")

    # the lists may be given as a name of a file in the project folder
    if not Path(args.obs_list).is_file():
        if (project_dir / args.obs_list).is_file():
            args.obs_list = str(project_dir / args.obs_list)
    if not Path(args.obs_list).is_file():
        sys.exit(f"I cannot find the observation list {args.obs_list}")

    if args.tiles is not None:
        if not Path(args.tiles).is_file():
            if (project_dir / args.tiles).is_file():
                args.tiles = str(project_dir / args.tiles)
        if not Path(args.tiles).is_file():
            sys.exit(f"I cannot find the tile list {args.tiles}")

    tile_list = None
    tile_problems = []
    if args.tiles is not None:
        tile_list, tile_problems = read_tile_list(args.tiles, args.max_tile)

    checked_at = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    rows = [check_observation(project_dir, obsid, tile_list, checked_at) for obsid in read_obs_list(args.obs_list)]

    out_dir = project_dir / ("summary_" + Path(args.obs_list).stem)         # obs_list.txt gives the folder summary_obs_list
    out_dir.mkdir(exist_ok=True)

    with open(str(out_dir / "table.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_lists(project_dir, out_dir, rows, Path(args.obs_list).stem)

    warnings = find_warnings(rows, tile_list, args.tiles, tile_problems)
    (out_dir / "warnings.txt").write_text("".join(w + "\n" for w in warnings))

    summary = make_summary_text(args.obs_list, rows, warnings, out_dir, checked_at)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)

    next_counts = " ".join(f"{state}={number}" for state, number in sorted(Counter(row["state"] for row in rows).items()))
    with open(str(project_dir / "summary_history.log"), "a") as handle:
        handle.write(f"{checked_at} {args.obs_list} total={len(rows)} {next_counts}\n")


if __name__ == "__main__":
    main()
