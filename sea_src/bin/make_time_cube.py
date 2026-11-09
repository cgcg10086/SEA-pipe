#!/usr/bin/env python

''' make a cube out of a bunch of fits files '''

import sys
from glob import glob
from astropy.io import fits
from astropy.time import Time
import numpy as np

prefix = sys.argv[1]

obsid = int(prefix[0:10])

files = glob(prefix+"-t????-image.fits")
files = sorted(files)

date_obs_list = []     # the time of every frame, read from the header of its image

for ind, f in enumerate(files):
    hdu = fits.open(f)
    date_obs_list.append(hdu[0].header["DATE-OBS"])

    if ind == 0:
        cube = hdu[0].data.copy()
#        cube.resize([len(files),hdu[0].data.shape[0],hdu[0].data.shape[1]])
        cube.resize([len(files),hdu[0].data.shape[2],hdu[0].data.shape[3]])
    else:
#        cube[ind,:,:] = hdu[0].data
        cube[ind,:,:] = np.squeeze(hdu[0].data)
    hdu.close()

hdu = fits.open(files[0])

try:
    crval3 = hdu[0].header["FREQ"]
except KeyError:
    crval3 = hdu[0].header["CRVAL3"]

hdu[0].data = cube

# # What is timestep
# starttime = Time(hdu[0].header["DATE-OBS"], format='isot', scale='utc')
# hdu2 = fits.open(files[1])
# nexttime = Time(hdu2[0].header["DATE-OBS"], format='isot', scale='utc')
# hdu2.close()
# delta = nexttime - starttime
# delta.format = "sec"
# step = int(delta.value)

# The time of every frame, in GPS seconds
gps_times = Time(date_obs_list, format="isot", scale="utc").gps

# The usual time between two frames. The median is not changed by a few missing frames.
steps = np.diff(gps_times)
step = float(np.median(steps))

# A missing frame shows up as a step that is longer than usual. Say so in the log.
for ind in np.where(np.abs(steps - step) > 0.5)[0]:
    print("WARNING: %.1f s between frame %d and frame %d, but the usual step is %.1f s" % (steps[ind], ind, ind + 1, step), file=sys.stderr)

# Keep record of frequency
hdu[0].header["FREQ"] = crval3

# Replace third axis details
# TODO figure out how to replace CTYPE3 comment (currently says "Central frequency")
hdu[0].header["CTYPE3"] = "TIME"
hdu[0].header["CRPIX3"] = 1.0
hdu[0].header["CRVAL3"] = gps_times[0]     # GPS time of the first frame
hdu[0].header["OBSID"] = obsid
hdu[0].header["TIMEREF"] = "middle of each frame"
hdu[0].header["CDELT3"] = step
hdu[0].header["CUNIT3"] = "sec"

# Keep the time of every frame inside the cube file, as a table in a second extension called FRAMES
frames_table = fits.BinTableHDU.from_columns([
    fits.Column(name="FRAME", format="J", array=np.arange(len(files))),
    fits.Column(name="GPSTIME", format="D", array=gps_times),
    fits.Column(name="DATEOBS", format="30A", array=np.array(date_obs_list)),
    fits.Column(name="FILE", format="64A", array=np.array(files)),
], name="FRAMES")
fits.HDUList([hdu[0], frames_table]).writeto("{0}-cube.fits".format(prefix), overwrite=True)

# Do the variance imaging while we're here

std = np.nanstd(hdu[0].data, axis=0)
hdu[0].data = std
hdu.writeto("{0}-stdev.fits".format(prefix), overwrite=True)

# # example: to read the times later 
# from astropy.io import fits
# times = fits.open("1476960704_transient-cube.fits")["FRAMES"].data["GPSTIME"]