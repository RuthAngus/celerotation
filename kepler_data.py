from __future__ import print_function
import numpy as np
import pyfits
import glob
import os


def load_kepler_data(LC_DIR):
    """
    load and join quarters together.
    Takes a list of fits file names for a given star.
    Returns the concatenated arrays of time, flux and flux_err
    """
    fnames = sorted(glob.glob(os.path.join(LC_DIR, "*llc.fits")))
    hdulist = pyfits.open(fnames[0])
    t = hdulist[1].data
    time = t["TIME"]
    flux = t["PDCSAP_FLUX"]
    flux_err = t["PDCSAP_FLUX_ERR"]
    q = t["SAP_QUALITY"]
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
            (q == 0)
    x = time[m]
    med = np.median(flux[m])
    y = flux[m]/med - 1
    yerr = flux_err[m]/med

    sections = np.zeros(len(x), dtype=int)
    for i, fname in enumerate(fnames[1:]):
       hdulist = pyfits.open(fname)
       t = hdulist[1].data
       time = t["TIME"]
       flux = t["PDCSAP_FLUX"]
       flux_err = t["PDCSAP_FLUX_ERR"]
       q = t["SAP_QUALITY"]
       m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
               (q == 0)

       x = np.concatenate((x, time[m]))
       med = np.median(flux[m])
       y = np.concatenate((y, flux[m]/med - 1))
       yerr = np.concatenate((yerr, flux_err[m]/med))
       sections = np.concatenate((sections, np.zeros(len(time[m]), dtype=int)
                                  +i+1))

    inds = np.argsort(x)
    x, y, yerr, sections = x[inds], y[inds], yerr[inds], sections[inds]
    return sections, x, y, yerr
