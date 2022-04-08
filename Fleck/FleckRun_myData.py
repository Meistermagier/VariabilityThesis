## Imports
import os
import shutil
from sympy import Mul
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
import astropy.units as u
import healpy as hp
from lightkurve import search_lightcurve
from emcee import EnsembleSampler,backends
from multiprocessing import Pool
from corner import corner
from fleck import Star
from schwimmbad import MPIPool,MultiPool
import zipfile
import json
from astropy.time import Time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Name", help="Input ZIP Filename", type=str)
parser.add_argument("-n", "--Number", help="Number of Samples.", type=int, default=100)
parser.add_argument("-c","--Cores",help="Ammount of Cores used",type=int,default = 4)
args = parser.parse_args()

print("Start Fleck Run with File: ",args.Name)
print("Iterations: ",args.Number)

### Read in the flux and time values from TESS with Lightcurve
with zipfile.ZipFile(args.Name,"r") as ZIP:
    with ZIP.open("Data.json") as File:

        Data = json.load(File)
        File.close()

    with ZIP.open("Raw_Data.json") as File:

        Data_Raw = json.load(File)
        File.close()

    ZIP.close()

Sel = Data["Header"]["SelectedID"]
MASS = Data["Header"]["2MASSID"]
Flux_Raw = Data_Raw[Sel]
Flux = np.array(Data["Curves"]["PSF_FLUX_COR"])
T = np.array(Data["Curves"]["Time"])

Flux_offset = Flux + np.median(Flux_Raw)

time = T
flux = Flux_offset

notnanflux = ~np.isnan(flux)

flux = flux[notnanflux]
time = time[notnanflux]

flux /= np.mean(flux)


### Retrieve Period 1

best_period = Data["MainModes"]["Period"][0]*u.d

print("Best Period",best_period)

### Prepare Initial Conditions
u_ld = [0.46, 0.11] #Limb Darkening
contrast = 0.7      #Contrast of the starspot compared to the base Brightness of 1
phases = (time % best_period.value) / best_period.value
s = Star(contrast, u_ld, n_phases=len(time), rotation_period=best_period.value)

init_lons = np.array([0, 320, 100])
init_lats = [0, 20, 0]
init_rads = [0.01, 0.2, 0.3]

yerr = 0.002

init_p = np.concatenate([init_lons, init_lats, init_rads])



### Prepare Likelihood functions and probabilities

def log_likelihood(p):
    lons = p[0:3]
    lats = p[3:6]
    rads = p[6:9]

    lc = s.light_curve(lons[:, None] * u.deg, lats[:, None] * u.deg, rads[:, None],
                        inc_stellar=90*u.deg, times=time, time_ref=0)[:, 0]

    ret = - 0.5 * np.sum((lc/np.mean(lc) - flux)**2 / yerr**2)
    return ret

def log_prior(p):
    lons = p[0:3]
    lats = p[3:6]
    rads = p[6:9]

    if (np.all(rads < 1.) and np.all(rads > 0) and np.all(lats > -60) and
        np.all(lats < 60) and np.all(lons > 0) and np.all(lons < 360)):
        return 0
    return -np.inf

def log_probability(p):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(p)


### Start the Walker and the Sampler
if __name__ == "__main__":
    ndim = len(init_p)
    nwalkers = 5 * ndim
    nsteps = args.Number

    pos = []


    while len(pos) < nwalkers:
        trial = init_p + 0.01 * np.random.randn(ndim)
        lp = log_prior(trial)
        if np.isfinite(lp):
            pos.append(trial)

    #Create Backend to Save
    file = f"Runs/{MASS}.h5"
    FirstTime = not os.path.exists(file)
    backend = backends.HDFBackend(file)
    if FirstTime:
        backend.reset(nwalkers, ndim)

    with Pool(args.Cores) as pool:
        sampler = EnsembleSampler(nwalkers, ndim, log_probability,pool=pool,backend=backend)
        sampler.run_mcmc(pos, nsteps, progress=True)


    samples_burned_in = sampler.flatchain[len(sampler.flatchain)//2:, :]
    
    
    ### Plotting Samples
    fig, ax = plt.subplots(9, 9, figsize=(6, 6))
    labels = "lon0 lon1 lon2 lat0 lat1 lat2 rad0 rad1 rad2".split()
    corner(samples_burned_in, smooth=True, labels=labels,
           fig=fig);


    ### Plotting Coverage and Sample Curves
    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 3))

    for i in np.random.randint(0, len(samples_burned_in), size=50):

        trial = samples_burned_in[i, :]

        lons = trial[0:3]
        lats = trial[3:6]
        rads = trial[6:9]

        lc = s.light_curve(lons[:, None] * u.deg, lats[:, None] * u.deg, rads[:, None],
                        inc_stellar=90*u.deg, times=time, time_ref=0)[:, 0]
        ax2[0].plot(time, lc/lc.mean(), color='DodgerBlue', alpha=0.05)

    f_S = np.sum(samples_burned_in[:, -3:]**2 / 4, axis=1)

    ax2[1].hist(f_S, bins=25, histtype='step', lw=2, color='k', range=[0, 0.12], density=True)
    ax2[0].set(xlabel='BJD - 2454833', ylabel='Flux', xticks=[2230, 2233, 2236, 2239])
    ax2[1].set_xlabel('$f_S$')
    ax2[0].plot(time, flux, '.', ms=2, color='k', zorder=10)

    NSIDE = 2**10

    NPIX = hp.nside2npix(NSIDE)

    m = np.zeros(NPIX)

    np.random.seed(0)
    random_index = np.random.randint(samples_burned_in.shape[0]//2,
                                    samples_burned_in.shape[0])
    random_sample = samples_burned_in[random_index].reshape((3, 3)).T
    for lon, lat, rad in random_sample:
        t = np.radians(lat + 90)
        p = np.radians(lon)
        spot_vec = hp.ang2vec(t, p)
        ipix_spots = hp.query_disc(nside=NSIDE, vec=spot_vec, radius=rad)
        m[ipix_spots] = 0.7

    cmap = plt.cm.Greys
    cmap.set_under('w')

    plt.axes(ax2[2])
    hp.mollview(m, cbar=False, title="", cmap=cmap, hold=True,
                max=1.0, notext=True, flip='geo')
    hp.graticule(color='silver')

    fig.suptitle('V1298 Tau')

    for axis in ax2:
        for sp in ['right', 'top']:
            axis.spines[sp].set_visible(False)

    ### Save the Plots

    fig.savefig("Runs/Temp/Fig_1.png",dpi=300,bbox_inches="tight")
    fig2.savefig("Runs/Temp/Fig_2.png",dpi=300,bbox_inches="tight")

    ### ZIP the Files
    shutil.copy(f"Runs/{MASS}.h5",f"Runs/Temp/{MASS}.h5")
    shutil.make_archive(f"Runs/{MASS}", 'zip', "Runs/Temp")