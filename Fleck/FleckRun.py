## Imports
import os

from sympy import Mul
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
import astropy.units as u
#import healpy as hp
from lightkurve import search_lightcurve
from emcee import EnsembleSampler,backends
from multiprocessing import Pool
from corner import corner
from fleck import Star
from schwimmbad import MPIPool,MultiPool




### Read in the flux and time values from TESS with Lightcurve
coord = SkyCoord.from_name('V1298 Tau')
slcf = search_lightcurve(coord, mission='TESS')

lc = slcf.download_all()
pdcsap = lc.stitch()

time = pdcsap.time.jd
flux = pdcsap.flux

notnanflux = ~np.isnan(flux)

flux = flux[notnanflux]
time = time[notnanflux]

flux /= np.mean(flux)


### Calculate best Period with Lombscargle
periods = np.linspace(1, 5, 1000) * u.day
freqs = 1 / periods
powers = LombScargle(time * u.d, flux).power(freqs)
best_period = periods[powers.argmax()]
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
    return ret.value

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
    nsteps = 1000

    pos = []


    while len(pos) < nwalkers:
        trial = init_p + 0.01 * np.random.randn(ndim)
        lp = log_prior(trial)
        if np.isfinite(lp):
            pos.append(trial)

    #Create Backend to Save
    file = "Fleck/Backend_V1298_Tau.h5"
    backend = backends.HDFBackend(file)
    #backend.reset(nwalkers, ndim)

    with Pool() as pool:
        sampler = EnsembleSampler(nwalkers, ndim, log_probability,pool=pool,backend=backend)
        sampler.run_mcmc(pos, nsteps, progress=True)


    samples_burned_in = sampler.flatchain[len(sampler.flatchain)//2:, :]
    
    
    ### Plotting Samples
    fig, ax = plt.subplots(9, 9, figsize=(6, 6))
    labels = "lon0 lon1 lon2 lat0 lat1 lat2 rad0 rad1 rad2".split()
    corner(samples_burned_in, smooth=True, labels=labels,
           fig=fig);


    ### Plotting Coverage and Sample Curves
    fig2, ax2 = plt.subplots(1, 2, figsize=(6, 4))

    for i in np.random.randint(0, len(samples_burned_in), size=50):

        trial = samples_burned_in[i, :]

        lons = trial[0:3]
        lats = trial[3:6]
        rads = trial[6:9]

        lc = s.light_curve(lons[:, None] * u.deg, lats[:, None] * u.deg, rads[:, None],
                           inc_stellar=90*u.deg, times=time, time_ref=0)[:, 0]
        ax2[0].plot(time, lc/lc.mean(),"-",zorder=1)

    f_S = np.sum(samples_burned_in[:, -3:]**2 / 4, axis=1)

    ax2[1].hist(f_S, bins=25, histtype='step', lw=2, color='k', range=[0, 0.12], density=True)
    ax2[0].set(xlabel='BJD - 2454833', ylabel='Flux', xticks=[2230, 2233, 2236, 2239])
    ax2[1].set_xlabel('$f_S$')
    ax2[0].plot(time, flux, '.', ms=2, color='k', zorder=0,alpha=0.4)

    ### Show the Plots

    fig.show()
    fig2.show()
    input("Press any Key to abort")