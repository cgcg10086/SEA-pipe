#!/usr/bin/env python
# Script original developed by John Morgan (JM)
# Example invocation from JM
# make_imstack.py 1204233560_IPS --suffixes=image -n 207
import os, sys, psutil, datetime, logging, h5py, contextlib
import numpy as np
from astropy.time import Time
# from optparse import OptionParser #NB zeus does not have argparse!
from argparse import ArgumentParser
from astropy.io import fits

VERSION = "0.5"
#changes from 0.1: lzf instead of gzip, support new wsclean which does not have WSCTIMES and WSCTIMEE
#changes from 0.2 (layout and metadata, measured on the NIGHT cubes: see InspectCube.py; a draft 0.3 with frame-CENTRE times was replaced before use):
#  - /image is no longer compressed or shuffled: noise-like float16 does not compress (ratio 1.00), so lzf+shuffle only cost CPU.  /model (almost all zeros,
#    70x smaller) keeps lzf+shuffle.  --compress-image restores the old behaviour.
#  - chunks are 64 x 64 pixels x all time (606 kB for 74 frames), was 16 x 16 (38 kB: 40,000 chunks per 3200 x 3200 cube; a 256 x 256 tile read 256 of them, now 16).
#    The image size no longer has to be divisible by --stamp-size: the chunks at the edge are simply partial.
#  - the whole wsclean command is kept (it was reduced to its last 72-character line), as /header attributes HISTORY (rejoined) and HISTORY_CARDS (raw), and COMMENT likewise
#  - /image and /model say what they are (MODEL_SUBTRACTED: yes / no / unknown, read from the wsclean command)
#  - TIMES FOLLOW THE CONVENTION OF THE FITS CUBES: every time is the MIDDLE of its integration, and DATE-OBS is the middle of the FIRST integration:
#      /header DATE-OBS   the first frame's DATE-OBS, i.e. the centre of the first integration (it was the centre of the MIDDLE frame, which is what happened to be copied)
#      /timestamp         unchanged: wsclean's DATE-OBS of each frame, the centre of that integration (UTC, 0.1 s)
#      /gpstime           GPS seconds at the centre of each frame (the first equals the FITS cubes' CRVAL3)
#      time axis          NAXIS = 5, NAXIS5 = frames, CTYPE5 = TIME, CUNIT5 = s, CDELT5 = step, CRPIX5 = 1, CRVAL5 = gpstime[0], TIMEREF = 'middle of each frame'
#                         (frame i is centred on CRVAL5 + (i - 1) x CDELT5; the observation starts half a step before CRVAL5, at the obsid)
#      HEADER_FRAME says which frame's header was copied; its own DATE-OBS is kept as HEADER_FRAME_DATE_OBS
#    wsclean writes the centre (checked: the first frame of 1441747424 reads 21:23:28.0 = GPS 1441747426, the obsid is 1441747424, steps are 4 s); if your wsclean
#    writes the START of each step, pass --date-obs-is start and the times are shifted to the centres.  (Drafts 0.3 and 0.4 existed only briefly; 0.4 used the START.)
#  - fixed: --verbose never took effect (logging.basicConfig was called twice); the header frame is start + n//2 (it was n//2, wrong or missing with --start > 0);
#    the file is closed explicitly
CACHE_SIZE = 1024 #MB
N_PASS = 1
TIME_INTERVAL = 4
TIME_INDEX = 0
POLS = 'I'
STAMP_SIZE = 64
SLICE = [0, 0, slice(None, None, None), slice(None, None, None)]
HDU = 0
PB_THRESHOLD = 0.1 # fraction of pbmax
#SUFFIXES=["image", "model", "dirty"]
SUFFIXES = "image,model"
N_TIMESTEPS = 25
N_CHANNELS = 1
DTYPE = np.float16
COMPRESSED_SUFFIXES = ("model",)    # sparse: lzf+shuffle makes it 70x smaller.  Everything else (the image) is noise-like and is stored as it is.
FILENAME = "{prefix}-t{time:04d}-{suffix}.fits"
PB_FILE = "{prefix}-beam.fits"

parser = ArgumentParser(
    description="Convert a set of wsclean images into an hdf5 image cube"
)
parser.add_argument('prefix', type=str, help='The prefix of file names to dump into the hdf5 file')
# parser.add_argument("-n", default=N_TIMESTEPS, type=int, help="number of timesteps to process [default: %default]")
# parser.add_argument("--start", default=TIME_INDEX,  type=int, help="starting time index [default: %default]")
# parser.add_argument("--step", default=TIME_INTERVAL, type=float, help="time between timesteps [default: %default]")
parser.add_argument("-n", default=N_TIMESTEPS, type=int, help="number of timesteps to process [default: %(default)s]")
parser.add_argument("--start", default=TIME_INDEX,  type=int, help="starting time index [default: %(default)s]")
parser.add_argument("--step", default=TIME_INTERVAL, type=float, help="time between timesteps [default: %(default)s]")

parser.add_argument("--outfile", default=None, type=str, help="outfile [default: [prefix].hdf5]")
# parser.add_argument("--suffixes", default=SUFFIXES, type=str, help="comma-separated list of suffixes to store [default: %default]")
# parser.add_argument("--stamp-size", default=STAMP_SIZE, type=int, help="hdf5 stamp size [default: %default]")
parser.add_argument("--suffixes", default=SUFFIXES, type=str, help="comma-separated list of suffixes to store [default: %(default)s]")
parser.add_argument("--stamp-size", default=STAMP_SIZE, type=int, help="hdf5 chunk size in pixels (chunks are stamp x stamp x all time); 64 reads a 256 x 256 tile as 16 chunks [default: %(default)s]")

parser.add_argument("--compress-image", action="store_true",  help="also compress the image-like datasets with lzf+shuffle (the 0.2 behaviour: no smaller, slower to read)")
parser.add_argument("--date-obs-is", choices=["centre", "start"], default="centre", help="what wsclean's per-frame DATE-OBS is: the CENTRE of the time step (wsclean 3.x, checked against the obsid) or its START, in which case the times are moved to the centres [default: %(default)s]")
parser.add_argument("--check-filenames-only", action="store_true",  help="check all required files are present then quit.")
parser.add_argument("--allow-missing", action="store_true",  help="check for presence of files for contiguous timesteps from --start up to -n")
parser.add_argument('-v','--verbose', action='store_true', default=False, help='More output logging')

args = parser.parse_args()

prefix = args.prefix

logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG if args.verbose else logging.INFO)

if args.outfile is None:
    args.outfile = "%s.hdf5" % prefix

if os.path.exists(args.outfile):
    logging.warning("Warning: editing existing file")
    file_mode = "r+"
else:
    file_mode = "w"

args.suffixes = args.suffixes.split(',')

# check that all image files are present
for suffix in args.suffixes:
    for t in range(args.start, args.n+args.start):
        infile = FILENAME.format(prefix=prefix, time=t, suffix=suffix)
        if not os.path.exists(infile):
            if args.allow_missing:
                new_n = t-args.start
                logging.info("couldn't find file %s: reducing n from %d to %d", infile, args.n, new_n)
                args.n = new_n
                break
            raise IOError("couldn't find file %s" % infile)
        logging.debug("%s found", infile)

if args.check_filenames_only:
    sys.exit()

propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
settings = list(propfaid.get_cache())
settings[2] *= CACHE_SIZE
propfaid.set_cache(*settings)

with contextlib.closing(h5py.h5f.create(args.outfile.encode("utf-8"), fapl=propfaid)) as fid:
    df = h5py.File(fid, file_mode)

df.attrs['VERSION'] = VERSION
df.attrs['USER'] = os.environ['USER']
df.attrs['DATE_CREATED'] = datetime.datetime.utcnow().isoformat()

group = df['/']
group.attrs['TIME_INTERVAL'] = args.step

# determine data size and structure
image_file = FILENAME.format(prefix=prefix, time=args.start, suffix=args.suffixes[0])
hdus = fits.open(image_file, memmap=True)
image_size = hdus[HDU].data.shape[-1]
args.stamp_size = min(args.stamp_size, image_size)
if image_size % args.stamp_size:
    logging.info("image size %d is not a multiple of the chunk size %d: the chunks at the edge are partial", image_size, args.stamp_size)
data_shape = [1, image_size, image_size, N_CHANNELS, args.n] # first axis is for polarisations
logging.debug("data shape: %s" % data_shape)
chunks = (1, args.stamp_size, args.stamp_size, N_CHANNELS, args.n)

pb_mask = np.ones(data_shape[1:-1] + [1], dtype=bool)
pb_nan = np.ones(data_shape[1:-1] + [1])

# write main header information
timestep_start = group.create_dataset("timestep_start", (args.n,), dtype=np.uint16)
timestep_stop = group.create_dataset("timestep_stop", (args.n,), dtype=np.uint16)
timestamp = group.create_dataset("timestamp", (args.n,), dtype="S21")
header_frame = args.start + args.n//2
header_file = FILENAME.format(prefix=prefix, time=header_frame, suffix=args.suffixes[0])


def commentary_text(fits_header, keyword):
    """All the HISTORY (or COMMENT) cards as one string.  wsclean writes its command as ONE long line, wrapped at the 72 characters of a card, in the middle of
    words, so a full card continues on the next and the cards are joined with nothing between them; a card that stops short is the end of a line."""
    out = []
    for card in fits_header.cards:
        if card.keyword == keyword:
            text = card.image[8:80]
            out.append(text if len(text.rstrip()) == 72 else text.rstrip() + "\n")
    return "".join(out).rstrip("\n")


# add fits header to attributes
hdus = fits.open(header_file, memmap=True)
header = group.create_dataset('header', data=[], dtype=DTYPE)
for key, item in hdus[0].header.items():
    if key not in ("HISTORY", "COMMENT", ""):             # repeated keywords: an attribute holds one value, so the old code kept only the LAST card of each
        header.attrs[key] = item
wsclean_command = commentary_text(hdus[0].header, "HISTORY")
header.attrs["HISTORY"] = wsclean_command
header.attrs["COMMENT"] = commentary_text(hdus[0].header, "COMMENT")
header.attrs["HISTORY_CARDS"] = np.array([c.image[8:80].rstrip() for c in hdus[0].header.cards if c.keyword == "HISTORY"], dtype=h5py.string_dtype())
header.attrs["HEADER_FRAME"] = header_frame

for s, suffix in enumerate(args.suffixes):
    logging.info("processing suffix %s" % (suffix))
    logging.info("about to allocate data %s", psutil.virtual_memory())

    if s == 0:
        data = np.zeros(data_shape, dtype=DTYPE)
    else:
        data *= 0
    filenames = group.create_dataset("%s_filenames" % suffix, (1, N_CHANNELS, args.n), dtype="S%d" % len(header_file), compression='lzf')

    n_rows = image_size
    i=0
    for t in range(args.n):
        im_slice = [slice(n_rows*i, n_rows*(i+1)), slice(None, None, None)]
        fits_slice = tuple(SLICE[:-2] + im_slice)

        infile = FILENAME.format(prefix=prefix, time=t+args.start, suffix=suffix)
        logging.info(" processing %s", infile)
        with fits.open(infile, memmap=True) as hdus:
            filenames[0, 0, t] = infile.encode("utf-8")
            data[0, n_rows*i:n_rows*(i+1), :, 0, t] = np.where(pb_mask[n_rows*i:n_rows*(i+1), :, 0, 0],
                                                            hdus[0].data[fits_slice],
                                                            np.nan)*pb_nan[n_rows*i:n_rows*(i+1), :, 0, 0]
            if s == 0:
                timestamp[t] = hdus[0].header['DATE-OBS'].encode("utf-8")
                timestep_start[t] = t
                timestep_stop[t] = t+1
            else:
                assert timestamp[t] == hdus[0].header['DATE-OBS'].encode("utf-8"), "Timesteps do not match %s in %s" % (args.suffixes[0], infile)
    logging.info(" writing to hdf5 file")
    compress = suffix in COMPRESSED_SUFFIXES or args.compress_image
    hdf5_data = group.create_dataset(suffix, data_shape, chunks=chunks, dtype=DTYPE, **(dict(compression='lzf', shuffle=True) if compress else {}))
    hdf5_data[...] = data
    if suffix == "model":
        hdf5_data.attrs["DESCRIPTION"] = "wsclean model image of each time step"
    else:
        sub_flag = ("yes" if "-subtract-model" in wsclean_command else "no") if wsclean_command else "unknown"
        hdf5_data.attrs["MODEL_SUBTRACTED"] = sub_flag
        hdf5_data.attrs["DESCRIPTION"] = "wsclean '%s' image of each time step; model subtracted: %s (from '-subtract-model' in the wsclean command in /header HISTORY)" % (suffix, sub_flag)
    logging.info(" done with %s (%s)" % (suffix, "lzf+shuffle" if compress else "uncompressed"))

# Frame times.  The convention (the same as the FITS cubes): every time is the MIDDLE of its integration and DATE-OBS is the middle of the FIRST integration.
# wsclean already writes the centre, so /timestamp stays exactly as wsclean wrote it; with --date-obs-is start the times are moved to the centres.
wsclean_stamps = [x.decode("utf-8") for x in timestamp[...]]
shift = 0.5 * args.step if args.date_obs_is == "start" else 0.0
gps_str = Time(wsclean_stamps, format="isot", scale="utc").gps + shift             # GPS seconds at the centre of each frame, from the (0.1 s truncated) strings
gps_arith = gps_str[0] + args.step * np.arange(args.n)
if np.max(np.abs(gps_str - gps_arith)) <= 0.15:                                    # equal steps are then the better estimate (the strings are truncated to 0.1 s)
    gps, how = gps_arith, "DATE-OBS of the first frame%s + TIME_INTERVAL x frame" % (" plus half a step (wsclean wrote the start)" if shift else "")
else:
    gps, how = gps_str, "from each frame's DATE-OBS (the frames are NOT %g s apart, so equal steps were not assumed)" % args.step
    logging.warning("the frames are not %g s apart: /gpstime comes from each frame's DATE-OBS", args.step)

if shift:                                                                          # wsclean wrote starts: /timestamp becomes the centres
    t_c = Time(gps, format="gps").utc
    t_c.precision = 1
    timestamp[...] = np.array(t_c.isot, dtype="S21")
timestamp.attrs["DESCRIPTION"] = "UTC middle of each integration (ISO 8601, 0.1 s), as in the FITS cubes"
gpstime = group.create_dataset("gpstime", data=np.asarray(gps, dtype=np.float64))
gpstime.attrs["UNITS"], gpstime.attrs["TIMEREF"], gpstime.attrs["FRAME_LENGTH"], gpstime.attrs["METHOD"] = "s (GPS)", "middle of each frame", float(args.step), how
if "DATE-OBS" in header.attrs:
    header.attrs["HEADER_FRAME_DATE_OBS"] = str(header.attrs["DATE-OBS"])           # the copied frame's own (wsclean) DATE-OBS
header.attrs.update({"DATE-OBS": timestamp[0].decode("utf-8"),                     # the middle of the first integration, as in the FITS cubes
                     "NAXIS": 5, "NAXIS5": args.n, "CTYPE5": "TIME", "CUNIT5": "s", "CDELT5": float(args.step), "CRPIX5": 1.0, "CRVAL5": float(gps[0]),
                     "TIMEREF": "middle of each frame",
                     "TIMEAXIS": "CRVAL5 = GPS seconds at the MIDDLE of frame 1 (CRPIX5 = 1); frame i is centred on CRVAL5 + (i - 1) x CDELT5; DATE-OBS is the same instant in UTC"})
df.close()