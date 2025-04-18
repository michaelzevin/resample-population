import numpy as np
import pandas as pd

import os
import h5py
import pdb
from tqdm import tqdm
import itertools

from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.stats import truncnorm

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.cosmology import z_at_value

from utils import filters


# --- SFR and metallicity functions --- #
def sfr_z(z, mdl='2017'):
    """
    Star formation rate as a function in redshift, in units of M_sun / Mpc^3 / yr
    mdl='2017': Default, from Madau & Fragos 2017. Tassos added more X-ray binaries at higher Z, brings rates down
    mdl='2014': From Madau & Dickenson 2014, used in Belczynski et al. 2016
        """
    if mdl=='2017':
        return 0.01*(1+z)**(2.6)  / (1+((1+z)/3.2)**6.2)
    if mdl=='2014':
        return 0.015*(1+z)**(2.7) / (1+((1+z)/2.9)**5.6)

def mean_metal_z(z, Zsun=0.017):
    """
    Mean (mass-weighted) metallicity as a function of redshift
    From Madau & Fragos 2017

    Returns [O/H] metallicity (in absolute units)
    """
    log_Z_Zsun = 0.153 - 0.074 * z**(1.34)
    return 10**(log_Z_Zsun) * Zsun

def metal_disp_truncnorm(z, sigmaZ, Zlow, Zhigh):
    """
    Gives a weight for each metallicity Z at a redshift of z by assuming
    the metallicities are log-normally distributed about Z

    Metallicities are in standard units ([Fe/H])
    Default dispersion is half a dex

    lowZ and highZ indicate the lower and upper metallicity bounds; values drawn below these
    will be reflected to ensure the distribution is properly normalized

    NOTE: Be careful in calculating the mean of a log-normal distribution correctly!
    """
    log_mean_Z = np.log10(mean_metal_z(z)) - (np.log(10)/2)*sigmaZ**2

    a, b = (np.log10(Zlow) - log_mean_Z) / sigmaZ, (np.log10(Zhigh) - log_mean_Z) / sigmaZ
    Z_dist = truncnorm(a, b, loc=log_mean_Z, scale=sigmaZ)

    return Z_dist

def corrected_means_for_truncated_lognormal(sigmaZ, Zlow, Zhigh):
    """
    Function that returns an interpolant to get an adjusted log-normal mean
    such that the resultant truncated normal distribution preserves the mean

    The interpolant will take in the log-normal mean that you *want* after truncation,
    and gives you the mean you should use when constructing your truncated normal distribution
    """
    log_desired_means = np.linspace(-5,1, 1000)   # tunable, eventually range will give bogus values
    means_for_constructing_lognormal = log_desired_means - (np.log(10)/2)*sigmaZ**2

    means_from_truncated_lognormal = []
    for m in means_for_constructing_lognormal:
        a, b = (np.log10(Zlow) - m) / sigmaZ, (np.log10(Zhigh) - m) / sigmaZ
        Z_dist = truncnorm(a, b, loc=m, scale=sigmaZ)
        means_from_truncated_lognormal.append(Z_dist.moment(1))

    truncated_mean_to_gaussian_mean = interp1d(means_from_truncated_lognormal, log_desired_means, \
                   bounds_error=False, fill_value=(np.min(log_desired_means), np.max(log_desired_means)))

    return truncated_mean_to_gaussian_mean

def metal_disp_truncnorm_corrected(z, mean_transformation_interp, sigmaZ, Zlow, Zhigh):
    """
    Gives the probability density function for the metallicity distribution at a given redshift
    using a 'corrected' mean to reproduce the mean of the Z(z) relation

    NOTE: Be careful in calculating the mean of a log-normal distribution correctly!
    """
    log_desired_mean_Z = np.log10(mean_metal_z(z))
    corrected_mean = mean_transformation_interp(log_desired_mean_Z)

    corrected_mean_for_truncated_lognormal = corrected_mean - (np.log(10)/2)*sigmaZ**2

    a, b = (np.log10(Zlow) - corrected_mean_for_truncated_lognormal) / sigmaZ, (np.log10(Zhigh) - corrected_mean_for_truncated_lognormal) / sigmaZ
    Z_dist = truncnorm(a, b, loc=corrected_mean_for_truncated_lognormal, scale=sigmaZ)

    return Z_dist


# --- Star formation history class --- #
class StarFormationHistory:
    """
    Class for determining birth redshifts and metallicities for a population of mergers
    """
    def __init__(self, method, sigmaZ, zmin, zmax, Zmin, Zmax, cosmo_path=None, verbose=False):
        # Instantiates the SFH class that will be used to draw redshifts and metallicities
        self.method = method
        self.zmin = zmin
        self.zmax = zmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        if method != 'illustris':
            self.sigmaZ = sigmaZ

        ### special treament for reading in Illustris data
        if method=='illustris':
            with h5py.File(cosmo_path, 'r') as f:
                time_bins = f['xedges'][:]   # years
                met_bins = f['yedges'][1:-1]   # kill the top and bottom metallicity bins, we won't need them and they go to inf
                Mform = f['mass'][:,1:-1]   # Msun

            # we only care about stuff before today
            tmax = cosmo.age(0).to(u.yr).value
            too_old_idx = np.argwhere(time_bins > tmax)[0][0]
            time_bins = np.squeeze(time_bins[:too_old_idx])
            # add tH to end of time_bins, assume this bin just spans until today
            time_bins = np.append(time_bins, tmax)
            Mform = np.squeeze(Mform[:too_old_idx, :])

            # get redshift bins based on the cut time bins above
            redshift_bins = []
            if verbose:
                print("    determining bin edges in redshift space")
            for t in tqdm(time_bins[:-1]):
                redshift_bins.append(z_at_value(cosmo.age, t*u.yr))
            redshift_bins.append(0)   # special treatment for most local bin
            redshift_bins = np.asarray(redshift_bins)

            # now, slice arrays based on zmin and zmax, if specified
            if (zmin is not None) and (zmin > 0):
                too_old_idx = np.argwhere(redshift_bins < zmin)[0][0]
                redshift_bins = np.squeeze(redshift_bins[:too_old_idx])
                time_bins = np.squeeze(time_bins[:too_old_idx])
                # add lowest redshift to end of array
                redshift_bins = np.append(redshift_bins, zmin)
                time_bins = np.append(time_bins, cosmo.age(zmin).to(u.yr).value)
                Mform = np.squeeze(Mform[:too_old_idx, :])
            if (zmax is not None) and (zmax < redshift_bins.max()):
                too_young_idx = np.argwhere(redshift_bins > zmax)[-1][0]
                redshift_bins = np.squeeze(redshift_bins[(too_young_idx+1):])
                time_bins = np.squeeze(time_bins[(too_young_idx+1):])
                # add highest redshift to end of array
                redshift_bins = np.insert(redshift_bins, 0, zmax)
                time_bins = np.insert(time_bins, 0, cosmo.age(zmax).to(u.yr).value)
                Mform = np.squeeze(Mform[(too_young_idx):, :])

            # calculate dt for each time bin
            dt = time_bins[1:] - time_bins[:-1]

            # slice arrays based on Zmin and Zmax, if specified
            if (Zmin is not None) and (Zmin > met_bins.min()):
                # we only care about stuff above some threshold metallicity
                metal_enough_idx = np.argwhere(met_bins >= Zmin)[0][0]
                # first, save sfr below Zmin
                bins_below_Zmin = np.squeeze(met_bins[:metal_enough_idx])
                Mform_below_Zmin = np.squeeze(Mform[:, :(metal_enough_idx-1)])
                sfr_pts_below_Zmin = np.sum(Mform_below_Zmin, axis=1) / dt / 100**3   # simulation from 100 Mpc^3 bin
                # now that we have this saved, truncate metallicity bins
                met_bins = np.squeeze(met_bins[metal_enough_idx:])
                # add lowest metallicity to beginning of met bins
                met_bins = np.insert(met_bins, 0, Zmin)
                Mform = np.squeeze(Mform[:, (metal_enough_idx-1):])
            if (Zmax is not None) and (Zmax < met_bins.max()):
                # and remove things that have metallicities above ~2Zsun
                too_metal_idx = np.argwhere(met_bins > Zmax)[0][0]
                # first, save sfr below Zmin
                bins_above_Zmax = np.squeeze(met_bins[too_metal_idx:])
                Mform_above_Zmax = np.squeeze(Mform[:, too_metal_idx:])
                sfr_pts_above_Zmax = np.sum(Mform_above_Zmax, axis=1) / dt / 100**3   # simulation from 100 Mpc^3 bin
                # now that we have this saved, truncate metallicity bins
                met_bins = np.squeeze(met_bins[:too_metal_idx])
                # add highest metallicity to end of array
                met_bins = np.append(met_bins, Zmax)
                Mform = np.squeeze(Mform[:, :too_metal_idx])

            # Make sure everything has the correct dimensions
            assert (Mform.shape[0]==len(time_bins)-1) and (Mform.shape[0]==len(redshift_bins)-1) and (Mform.shape[1]==len(met_bins)-1), "Dimensions of bin heights in Mform ({:d,:d}) do not match the time ({:d}), redshift ({:d}), and metallicity ({:d}) bins!".format(Mform.shape[0], Mform.shape[1], len(time_bins), len(redshift_bins), len(met_bins))

            # get times and metallicity bin centers (in logspace)
            times = 10**(np.log10(time_bins[:-1])+((np.log10(time_bins[1:])-np.log10(time_bins[:-1]))/2))
            mets = 10**(np.log10(met_bins[:-1])+((np.log10(met_bins[1:])-np.log10(met_bins[:-1]))/2))

            # calculate redshifts for the log-spaced centers (for interpolation)
            redshifts = []
            if verbose:
                print("    determining bin centers in redshift space")
            for t in tqdm(times):
                redshifts.append(z_at_value(cosmo.age, t*u.yr))
            redshifts = np.asarray(redshifts)

            # get total SFRD at each redshift bin (units Msun / Mpc^3 / yr)
            sfr_pts = np.sum(Mform, axis=1) / dt / 100**3   # simulation from 100 Mpc^3 bin

            # pre-compute the metallicity CDF interpolations at each redshift bin
            met_cdf_at_redz = []
            for met_dist in Mform:
                if np.sum(met_dist) != 0.0:
                    met_cdf = cumulative_trapezoid(met_dist, np.log10(mets), initial=0)
                    met_cdf /= met_cdf.max()
                    met_cdf_interp = interp1d(np.log10(mets), met_cdf)
                    met_cdf_at_redz.append(met_cdf_interp)
                else:
                    # no SF at this time in any metallicity bin
                    met_cdf_at_redz.append(np.nan)

            # store as attributes and rearrange so that low redshifts are at the front
            self.Mform = Mform[::-1]
            self.times = times[::-1]
            self.mets = mets
            self.met_cdf_at_redz = met_cdf_at_redz[::-1]
            self.redshifts = redshifts[::-1]
            self.dt = dt[::-1]
            self.sfr_pts = sfr_pts[::-1]
            self.time_bins = time_bins[::-1]
            self.met_bins = met_bins
            self.redshift_bins = redshift_bins[::-1]
            # store SFR below Zmin and above Zmax for normalization purposes
            self.sfr_below_Zmin = sfr_pts_below_Zmin[::-1]
            self.sfr_above_Zmax = sfr_pts_above_Zmax[::-1]



    # --- Weights for each redshift and metallicity --- #
    def redshift_metallicity_weights(self, pop_path, N_redshift_grid=1e5, verbose=False):
        """
        Determines formation efficiencies (\gamma(z,Z)) and
        metallicity weights (dP/dz(z,Z)) for the discrete
        metallicities of the sim over a range of redshifts
        """
        # get redshift grid and create interpolant from lookback time to redshift (tlb in Myr)
        redz_grid = np.linspace(0, self.zmax, int(N_redshift_grid))
        tlb_grid = cosmo.lookback_time(redz_grid).to(u.Myr).value
        tlb_to_redz_interp = interp1d(tlb_grid, redz_grid)
        self.tlb_to_redz_interp = tlb_to_redz_interp
        self.redz_grid = redz_grid
        # set spacing in equal lookback times (e.g., 10 Myr)...do we need to account for time dilation? relative volume?

        # read metallicity of the population models
        hfile = h5py.File(pop_path, 'r')
        pop_mets = np.sort([float(x) for x in hfile.keys()])
        pop_mets_str = [str(x) for x in pop_mets]
        self.pop_mets = pop_mets
        self.pop_mets_str = pop_mets_str
        if (pop_mets.min() < self.Zmin):
            raise ValueError("Your low metallicity cutoff ({:0.1e}) is above your lowest metallicity run ({:0.1e})!".format(self.Zmin, pop_mets.min()))
        if (pop_mets.max() > self.Zmax):
            raise ValueError("Your high metallicity cutoff ({:0.1e}) is below your highest metallicity run ({:0.1e})!".format(self.Zmax, pop_mets.max()))

        # get total mass sampled, stored as attributes
        mass_sampled = []
        for Z in pop_mets_str:
            mass_sampled.append(float(hfile[Z].attrs['mass_sampled']))
        hfile.close()

        # get target population formation efficiency for each metallicity after applying filters
        formation_efficiencies = []
        if verbose:
            print("    determining formation efficiencies of population")
        for Z,Msamp in tqdm(zip(pop_mets_str, mass_sampled), total=len(pop_mets_str)):
            df = pd.read_hdf(pop_path, key=Z)
            formation_efficiencies.append(len(df) / Msamp)
        formation_efficiencies = np.asarray(formation_efficiencies)

        # create empty array for storing redshift and metallicity weights
        weight_array = np.empty((len(redz_grid), len(pop_mets)))

        # cycle over redshifts and metallicities to determine sampling weights at each discrete value
        if verbose:
            print("    determine relative weights at each redshift/metallicity bin")
        for idx, redz in tqdm(enumerate(redz_grid), total=len(redz_grid)):
            if self.method=='illustris':
                # we do try/except method here since bins with no star formation have NaNs instead of CDFs
                try:
                    # find the correct redshift bin (special treatment for most local bin)
                    redshift_idx = 0 if redz==0 else np.argwhere(self.redshift_bins < redz)[-1][0]
                    dt = tlb_grid[redshift_idx+1] - tlb_grid[redshift_idx]
                    # get the relative SFR contribution at this redshift
                    SFRD_contribution = self.sfr_pts[redshift_idx]
                    # grab metallicity distribution at this redshift
                    met_cdf = self.met_cdf_at_redz[redshift_idx]
                    # evaluate metallicity grid in the CDF
                    met_cdf_eval = met_cdf(np.log10(pop_mets))
                    # take the midpoints between CDF evaluations and assign weights to each metallicity
                    met_cdf_eval = met_cdf_eval[:-1] + (met_cdf_eval[1:]-met_cdf_eval[:-1])/2
                    met_cdf_eval = np.concatenate([[0],met_cdf_eval,[1]])
                    met_weights = met_cdf_eval[1:] - met_cdf_eval[:-1]
                    # make sure weights are not negative (can happen from precision error)
                    met_weights[np.argwhere(met_weights < 0)] = 0
                    assert np.round(np.sum(met_weights), 5)==1, "Metallicity weights at redshift {:0.3e} do not add to unity!".format(redz)
                except:
                    met_weights = np.zeros_like(pop_mets)
                    SFRD_contribution = 0.0
            else:
                # get the relative SFR contribution
                SFRD_contribution = sfr_z(redz)
                # grab the metallicity distribution at this redshift
                if self.method=='truncnorm':
                    met_dist = metal_disp_truncnorm(redz, self.sigmaZ, self.Zmin, self.Zmax)
                elif self.method=='corrected_truncnorm':
                    truncated_mean_to_gaussian_mean = corrected_means_for_truncated_lognormal(self.sigmaZ, self.Zmin, self.Zmax)
                    met_dist = metal_disp_truncnorm_corrected(redz, truncated_mean_to_gaussian_mean, self.sigmaZ, self.Zmin, self.Zmax)
                # evaluate metallicity grid in the CDF
                met_cdf_eval = met_dist.cdf(np.log10(pop_mets))
                # take the midpoints between CDF evaluations and assign weights to each metallicity
                met_cdf_eval = met_cdf_eval[:-1] + (met_cdf_eval[1:]-met_cdf_eval[:-1])/2
                met_cdf_eval = np.concatenate([[0],met_cdf_eval,[1]])
                met_weights = met_cdf_eval[1:] - met_cdf_eval[:-1]
                # make sure weights are not negative (can happen from precision error)
                met_weights[np.argwhere(met_weights < 0)] = 0
                assert np.round(np.sum(met_weights), 5)==1, "Metallicity weights at redshift {:0.3e} do not add to unity!".format(redz)

            dtdz = (cosmo.H(redz).value * (1+redz))**(-1)
            weight_array[idx] = SFRD_contribution * formation_efficiencies * met_weights * dtdz

        weight_array /= np.sum(weight_array)
        self.weight_array = weight_array



    # --- Resampling method --- #
    def resample(self, N, pop_path, mergers_only=False, verbose=False):
        """
        Resamples population. Metallicities and efficiencies are required, and
        will by default return all parameters in provided dataframe
        """
        # randomly choose redshifts and metallicities based on the weight array
        redz_met_grid = np.asarray(list(itertools.product(self.redz_grid, self.pop_mets)))
        redz_met_grid_idxs = np.arange(len(redz_met_grid))
        redz_met_draws = redz_met_grid[np.random.choice(redz_met_grid_idxs, N, p=self.weight_array.flatten())]
        redz_draws = redz_met_draws[:,0]
        met_draws = redz_met_draws[:,1]

        # create dataframe for housing the resampled data
        df = pd.DataFrame(np.asarray([redz_draws, met_draws]).T, columns=['z_zams','met'])
        df['tlb_zams'] = cosmo.lookback_time(np.asarray(df['z_zams'])).to(u.Myr).value

        if verbose:
            print("    selecting binaries from each discrete metallicity in population model")
        for idx, Z in tqdm(enumerate(self.pop_mets_str), total=len(self.pop_mets_str)):
            # if no samples were drawn from this metallicity, move on
            if len(df.loc[df['met']==self.pop_mets[idx]])==0:
                print("No systems were drawn for Z={:s}!".format(Z))
                continue

            # read in population model
            pop_model = pd.read_hdf(pop_path, key=Z)

            # if this is the first model, initialize the resampled dataframe
            if idx==0:
                for key in pop_model.keys():
                    df[key] = np.nan * np.ones(len(df))

            # randomly choose systems from this model
            idxs_in_metbin = np.asarray(df.loc[df['met']==self.pop_mets[idx]].index)
            idxs_to_sample = np.random.choice(list(pop_model.index), size=len(idxs_in_metbin), replace=True)
            resampled_pop = pop_model.loc[idxs_to_sample]
            for key in resampled_pop.keys():
                df.loc[idxs_in_metbin, key] = np.asarray(resampled_pop[key])

        # remove systems that merged after z=0, if specified
        if mergers_only==True:
            df = df.loc[df['t_merge'] < df['tlb_zams']]

        # Make sure there are no NaN values left
        assert(all(df.isna().any())==False)

        self.resampled_pop = df



    # --- Save to disk --- #
    def save(self, output_path):
        """
        Resamples populations living at `pop_path` based on the drawn formation redshifts and metallicities
        """
        self.resampled_pop.to_hdf(output_path, key='underlying')




    ########################################################
    ### RESAMPLING METHOD SPECIFICALLY FOR COSMIC OUTPUT ###
    ########################################################

    # --- Weights for each redshift and metallicity --- #
    def redshift_metallicity_weights_cosmic(self, pop_path, pop_filters=None, N_redshift_grid=1e5, verbose=False):
        """
        Determines formation efficiencies (\gamma(z,Z)) and
        metallicity weights (dP/dz(z,Z)) for the discrete
        metallicities of the sim over a range of redshifts
        """
        # get redshift grid and create interpolant from lookback time to redshift (tlb in Myr)
        redz_interp = np.linspace(0, self.zmax, 10000)
        tlb_interp = cosmo.lookback_time(redz_interp).to(u.Myr).value
        tlb_to_redz_interp = interp1d(tlb_interp, redz_interp)
        self.tlb_to_redz_interp = tlb_to_redz_interp

        # create grid of potential formation times, uniform in lookback time
        tlb_grid = np.linspace(cosmo.lookback_time(self.zmin), cosmo.lookback_time(self.zmax), int(N_redshift_grid)).to(u.Myr).value
        redz_grid = tlb_to_redz_interp(tlb_grid)
        self.redz_grid = redz_grid

        # read metallicity of the population models
        pop_mets = np.sort([float(x) for x in os.listdir(pop_path) if '0.' in x])
        pop_mets_str = sorted([x for x in os.listdir(pop_path) if '0.' in x])
        self.pop_mets = pop_mets
        self.pop_mets_str = pop_mets_str
        if (pop_mets.min() < self.Zmin):
            raise ValueError("Your low metallicity cutoff ({:0.1e}) is above your lowest metallicity run ({:0.1e})!".format(self.Zmin, pop_mets.min()))
        if (pop_mets.max() > self.Zmax):
            raise ValueError("Your high metallicity cutoff ({:0.1e}) is below your highest metallicity run ({:0.1e})!".format(self.Zmax, pop_mets.max()))

        # get target population formation efficiency for each metallicity after applying filters
        formation_efficiencies = []
        if verbose:
            print("    determining formation efficiencies after applying filters")
        for Z in tqdm(pop_mets_str):
            dat_file = [x for x in os.listdir(os.path.join(pop_path, Z)) if 'dat' in x][0]
            total_mass_sampled = float(np.asarray(pd.read_hdf(os.path.join(pop_path,Z,dat_file), key='mass_stars'))[-1])
            bpp = pd.read_hdf(os.path.join(pop_path,Z,dat_file), key='bpp')
            initC = pd.read_hdf(os.path.join(pop_path,Z,dat_file), key='initCond')
            # apply filters to bpp array, if specified
            if pop_filters is not None:
                for filt in pop_filters:
                    if filt in filters._filters_dict.keys():
                        bpp,initC = filters._filters_dict[filt](bpp,initC)
                    else:
                        raise ValueError('The filter you specified ({:s}) is not defined in the filters function!'.format(filt))
            # now get the formation efficiency of the target population
            N_form = len(bpp['bin_num'].unique())
            formation_efficiencies.append(N_form / total_mass_sampled)
        formation_efficiencies = np.asarray(formation_efficiencies)

        # create empty array for storing redshift and metallicity weights
        weight_array = np.empty((len(redz_grid), len(pop_mets)))

        # cycle over redshifts and metallicities to determine sampling weights at each discrete value
        if verbose:
            print("    determine relative weights at each redshift/metallicity bin")
        for idx, redz in tqdm(enumerate(redz_grid), total=len(redz_grid)):
            if self.method=='illustris':
                # we do try/except method here since bins with no star formation have NaNs instead of CDFs
                try:
                    # find the correct redshift bin (special treatment for most local bin)
                    redshift_idx = 0 if redz==0 else np.argwhere(self.redshift_bins < redz)[-1][0]
                    dt = tlb_grid[redshift_idx+1] - tlb_grid[redshift_idx]
                    # get the relative SFR contribution at this redshift
                    SFRD_contribution = self.sfr_pts[redshift_idx]
                    # grab metallicity distribution at this redshift
                    met_cdf = self.met_cdf_at_redz[redshift_idx]
                    # evaluate metallicity grid in the CDF
                    met_cdf_eval = met_cdf(np.log10(pop_mets))
                    # take the midpoints between CDF evaluations and assign weights to each metallicity
                    met_cdf_eval = met_cdf_eval[:-1] + (met_cdf_eval[1:]-met_cdf_eval[:-1])/2
                    met_cdf_eval = np.concatenate([[0],met_cdf_eval,[1]])
                    met_weights = met_cdf_eval[1:] - met_cdf_eval[:-1]
                    # make sure weights are not negative (can happen from precision error)
                    met_weights[np.argwhere(met_weights < 0)] = 0
                    assert np.round(np.sum(met_weights), 5)==1, "Metallicity weights at redshift {:0.3e} do not add to unity!".format(redz)
                except:
                    met_weights = np.zeros_like(pop_mets)
                    SFRD_contribution = 0.0
            else:
                # get the relative SFR contribution
                SFRD_contribution = sfr_z(redz)
                # grab the metallicity distribution at this redshift
                if self.method=='truncnorm':
                    met_dist = metal_disp_truncnorm(redz, self.sigmaZ, self.Zmin, self.Zmax)
                elif self.method=='corrected_truncnorm':
                    truncated_mean_to_gaussian_mean = corrected_means_for_truncated_lognormal(self.sigmaZ, self.Zmin, self.Zmax)
                    met_dist = metal_disp_truncnorm_corrected(redz, truncated_mean_to_gaussian_mean, self.sigmaZ, self.Zmin, self.Zmax)
                # evaluate metallicity grid in the CDF
                met_cdf_eval = met_dist.cdf(np.log10(pop_mets))
                # take the midpoints between CDF evaluations and assign weights to each metallicity
                met_cdf_eval = met_cdf_eval[:-1] + (met_cdf_eval[1:]-met_cdf_eval[:-1])/2
                met_cdf_eval = np.concatenate([[0],met_cdf_eval,[1]])
                met_weights = met_cdf_eval[1:] - met_cdf_eval[:-1]
                # make sure weights are not negative (can happen from precision error)
                met_weights[np.argwhere(met_weights < 0)] = 0
                assert np.round(np.sum(met_weights), 5)==1, "Metallicity weights at redshift {:0.3e} do not add to unity!".format(redz)

            dtdz = (cosmo.H(redz) * (1+redz))**(-1)
            weight_array[idx] = SFRD_contribution * formation_efficiencies * met_weights * dtdz

        weight_array /= np.sum(weight_array)
        self.weight_array = weight_array
        formation_efficiencies_dict = {}
        for met, f in zip(pop_mets, formation_efficiencies):
            formation_efficiencies_dict[met] = f
        self.formation_efficiencies = formation_efficiencies_dict


    # --- Resampling method for cosmic --- #
    def resample_cosmic(self, N, pop_path, pop_filters=None, mergers_only=False, initC_only=False, extra_info=False, verbose=False):
        """
        Resamples populations living at `pop_path`
        """
        # randomly choose redshifts and metallicities based on the weight array
        redz_met_grid = np.asarray(list(itertools.product(self.redz_grid, self.pop_mets)))
        redz_met_grid_idxs = np.arange(len(redz_met_grid))
        redz_met_draws = redz_met_grid[np.random.choice(redz_met_grid_idxs, N, p=self.weight_array.flatten())]
        redz_draws = redz_met_draws[:,0]
        met_draws = redz_met_draws[:,1]

        # create dataframe for housing the resampled data
        df = pd.DataFrame(np.asarray([redz_draws, met_draws]).T, columns=['z_ZAMS','Z'])
        df['tlb_ZAMS'] = cosmo.lookback_time(np.asarray(df['z_ZAMS'])).to(u.Myr).value

        if initC_only==False:
            # create empty columns for all the parameters we wish to save in the resampled population
            default_params = ['tlb_DCO','z_DCO','tlb_merge','z_merge','m1','m2','a','porb','e']
            extra_params = ['M1_ZAMS','M2_ZAMS','porb_ZAMS','e_ZAMS','secondary_born_first',\
                    'secondary_born_first','Mbh1_birth','Mbh1_preMT','Mbh1_postMT','Mstar_preMT',\
                    'Mstar_postMT','delta_t_MT','porb_HeBH','Mbh_HeBH','Mhe_HeBH','porb_HeHe',\
                    'Mhe1_HeHe','Mhe2_HeHe','Mbh1','Mbh2','SN_theta','evo_pathway']
            df_tmp = pd.DataFrame(np.nan*np.ones((len(df),len(default_params))), columns=default_params)
            df = pd.concat([df, df_tmp], axis=1)
            if extra_info==True:
                df_tmp = pd.DataFrame(np.nan*np.ones((len(df),len(extra_params))), columns=extra_params)
                df = pd.concat([df, df_tmp], axis=1)


        if verbose:
            print("    selecting binaries from each discrete metallicity in population model")
        for idx, Z in tqdm(enumerate(self.pop_mets_str), total=len(self.pop_mets_str)):
            # if no samples were drawn from this metallicity, move on
            if len(df.loc[df['Z']==self.pop_mets[idx]])==0:
                print("No systems were drawn for Z={:s}!".format(Z))
                continue

            # read in COSMIC data
            dat_file = [x for x in os.listdir(os.path.join(pop_path, Z)) if 'dat' in x][0]
            bpp = pd.read_hdf(os.path.join(pop_path, Z, dat_file), key='bpp')
            initC = pd.read_hdf(os.path.join(pop_path, Z, dat_file), key='initCond')
            mass_stars = float(np.asarray(pd.read_hdf(os.path.join(pop_path, Z, dat_file), key='mass_stars'))[-1])
            if initC_only==True and idx==0:
                # create empty columns for each initC line in the dataframe (only do this once)
                df_tmp = pd.DataFrame(np.nan*np.ones((len(df),len(initC.keys()))), columns=list(initC.keys()))
                df = pd.concat([df, df_tmp], axis=1)

            # apply filters to bpp array, if specified
            if pop_filters is not None:
                for filt in pop_filters:
                    if filt in filters._filters_dict.keys():
                        bpp,initC = filters._filters_dict[filt](bpp,initC)
                    else:
                        raise ValueError('The filter you specified ({:s}) is not defined in the filters function!'.format(filt))

            # if we only wish to save the initial conditions of the resampled population, do this here and skip everything else in the loop
            if initC_only==True:
                # randomly choose systems from this model
                idxs_in_metbin = np.asarray(df.loc[df['Z']==self.pop_mets[idx]].index)
                idxs_to_sample = np.random.choice(list(initC.index), size=len(idxs_in_metbin), replace=True)
                initC_sample = initC.loc[idxs_to_sample]
                for key in initC.keys():
                    df.loc[idxs_in_metbin, key] = np.asarray(initC_sample[key])
                continue

            # get point of DCO formation
            dco_form = bpp.loc[((bpp['kstar_1']==13)|(bpp['kstar_1']==14)) \
                               & ((bpp['kstar_2']==13)|(bpp['kstar_2']==14))].groupby('bin_num').first()
            dco_merge = bpp.loc[((bpp['kstar_1']==13)|(bpp['kstar_1']==14)) \
                               & ((bpp['kstar_2']==13)|(bpp['kstar_2']==14))].groupby('bin_num').last()

            # set the tphys of things that don't merge in a Hubble time to NaN so that tlb_merge will be NaN
            dco_merge.loc[dco_merge['evol_type']==10, 'tphys'] = np.nan

            # if there are no systems in the population model that satisfy the criteria, just leave all their entries and NaNs
            if (len(dco_form)==0) or (len(dco_merge)==0):
                print("No matching systems found in the population model for Z={:s}!".format(Z))
                continue

            # randomly choose systems from this model
            idxs_in_metbin = np.asarray(df.loc[df['Z']==self.pop_mets[idx]].index)
            idxs_to_sample = np.random.choice(list(dco_form.index), size=len(idxs_in_metbin), replace=True)
            dco_form_sample = dco_form.loc[idxs_to_sample]
            dco_merge_sample = dco_merge.loc[idxs_to_sample]

            # set these values in the dataframe
            df.loc[idxs_in_metbin, 'tlb_DCO'] = np.asarray(df.loc[idxs_in_metbin, 'tlb_ZAMS']) - np.asarray(dco_form_sample['tphys'])
            df_tmp = df.loc[idxs_in_metbin]   # Needed to get indices for things that merge after z=0
            idxs_in_metbin_tH = list(df_tmp.loc[df_tmp['tlb_DCO'] > 0].index)
            df.loc[idxs_in_metbin_tH, 'z_DCO'] = self.tlb_to_redz_interp(np.asarray(df.loc[idxs_in_metbin_tH, 'tlb_DCO']))

            df.loc[idxs_in_metbin, 'tlb_merge'] = np.asarray(df.loc[idxs_in_metbin, 'tlb_ZAMS']) - np.asarray(dco_merge_sample['tphys'])
            df_tmp = df.loc[idxs_in_metbin]   # Needed to get indices for things that merge after z=0
            idxs_in_metbin_tH = list(df_tmp.loc[df_tmp['tlb_merge'] > 0].index)
            df.loc[idxs_in_metbin_tH, 'z_merge'] = self.tlb_to_redz_interp(np.asarray(df.loc[idxs_in_metbin_tH, 'tlb_merge']))

            df.loc[idxs_in_metbin, 'm1'] = np.maximum(np.asarray(dco_form_sample['mass_1']), np.asarray(dco_form_sample['mass_2']))
            df.loc[idxs_in_metbin, 'm2'] = np.minimum(np.asarray(dco_form_sample['mass_1']), np.asarray(dco_form_sample['mass_2']))
            df.loc[idxs_in_metbin, 'a'] = np.asarray(dco_form_sample['sep'])
            df.loc[idxs_in_metbin, 'porb'] = np.asarray(dco_form_sample['porb'])
            df.loc[idxs_in_metbin, 'e'] = np.asarray(dco_form_sample['ecc'])

            # --- EXTRA INFO --- #
            # get extra info for determining spins, if specified
            if extra_info:
                # get ZAMS properties
                df.loc[idxs_in_metbin, 'M1_ZAMS'] = np.asarray(initC.loc[idxs_to_sample, 'mass_1'])
                df.loc[idxs_in_metbin, 'M2_ZAMS'] = np.asarray(initC.loc[idxs_to_sample, 'mass_2'])
                df.loc[idxs_in_metbin, 'porb_ZAMS'] = np.asarray(initC.loc[idxs_to_sample, 'porb'])
                df.loc[idxs_in_metbin, 'e_ZAMS'] = np.asarray(initC.loc[idxs_to_sample, 'ecc'])
                # mark small fraction of systems where the secondary is born first
                df_firstborn = bpp.loc[((bpp.kstar_1==14) & (bpp.kstar_2<14)) | \
                        ((bpp.kstar_1<14) & (bpp.kstar_2==14))].groupby('bin_num').head(1)
                df_firstborn_sample = df_firstborn.loc[idxs_to_sample]
                # find where secondary was born first
                secondary_born_first = np.where(df_firstborn_sample['kstar_2']==14, True, False)
                df.loc[idxs_in_metbin, 'secondary_born_first'] = secondary_born_first
                df.loc[idxs_in_metbin[~secondary_born_first], 'Mbh1_birth'] = np.asarray(df_firstborn_sample[~secondary_born_first]['mass_1'])
                df.loc[idxs_in_metbin[secondary_born_first], 'Mbh1_birth'] = np.asarray(df_firstborn_sample[secondary_born_first]['mass_2'])

                # first-born BH mass before and after RLO (if SMT happens at all)
                rlo_start = bpp.loc[((bpp.kstar_1==14) & (bpp.kstar_2<14) & (bpp.evol_type==3))].groupby('bin_num').head(1)
                rlo_end = bpp.loc[((bpp.kstar_1==14) & (bpp.evol_type==4))].groupby('bin_num').tail(1)
                # get sampling indices that went through RLO, rest will just be NaNs for now
                idxs_with_RLO = pd.Series(idxs_to_sample).isin(rlo_start.index.unique())
                #idxs_with_RLO = pd.Series(idxs_with_RLO).isin(rlo_end.index.unique())   # redundant
                # now we should be good, since the mass is the same before/after RLO
                rlo_start_sample = rlo_start.loc[idxs_to_sample[idxs_with_RLO]]
                rlo_end_sample = rlo_end.loc[idxs_to_sample[idxs_with_RLO]]
                df.loc[idxs_in_metbin[idxs_with_RLO], 'Mbh1_preMT'] = np.asarray(rlo_start_sample['mass_1'])
                df.loc[idxs_in_metbin[idxs_with_RLO], 'Mbh1_postMT'] = np.asarray(rlo_end_sample['mass_1'])
                df.loc[idxs_in_metbin[idxs_with_RLO], 'Mstar_preMT'] = np.asarray(rlo_start_sample['mass_2'])
                df.loc[idxs_in_metbin[idxs_with_RLO], 'Mstar_postMT'] = np.asarray(rlo_end_sample['mass_2'])
                df.loc[idxs_in_metbin[idxs_with_RLO], 'delta_t_MT'] = np.asarray(rlo_end_sample['tphys'] - rlo_start_sample['tphys'])

                # get the last timestep of BH+HeMS for standard tidal spinup
                HeBH = bpp.loc[((bpp.kstar_1==14) & (bpp.kstar_2==7)) | \
                                    ((bpp.kstar_1==7) & (bpp.kstar_2==14))].groupby('bin_num').tail(1)
                porb, Mbh, Mhe = [], [], []
                for samp_idx,secondary_first in zip(idxs_to_sample,secondary_born_first):
                    if samp_idx in HeBH.index.unique():
                        porb.append(HeBH.loc[samp_idx, 'porb'])
                        if secondary_first==True:
                            Mbh.append(HeBH.loc[samp_idx, 'mass_2'])
                            Mhe.append(HeBH.loc[samp_idx, 'mass_1'])
                        else:
                            Mbh.append(HeBH.loc[samp_idx, 'mass_1'])
                            Mhe.append(HeBH.loc[samp_idx, 'mass_2'])
                    else:
                        porb.append(np.nan)
                        Mbh.append(np.nan)
                        Mhe.append(np.nan)
                df.loc[idxs_in_metbin, 'porb_HeBH'] = np.asarray(porb)
                df.loc[idxs_in_metbin, 'Mbh_HeBH'] = np.asarray(Mbh)
                df.loc[idxs_in_metbin, 'Mhe_HeBH'] = np.asarray(Mhe)

                # get the timestep when both stars are He stars for potential spinup of first-born in the doubleCE channel
                HeHe = bpp.loc[((bpp.kstar_1==7) & (bpp.kstar_2==7)) | \
                                    ((bpp.kstar_1==7) & (bpp.kstar_2==7))].groupby('bin_num').tail(1)
                porb, Mhe1, Mhe2 = [], [], []
                for samp_idx,secondary_first in zip(idxs_to_sample,secondary_born_first):
                    if samp_idx in HeHe.index.unique():
                        porb.append(HeHe.loc[samp_idx, 'porb'])
                        if secondary_first==True:
                            Mhe1.append(HeHe.loc[samp_idx, 'mass_2'])
                            Mhe2.append(HeHe.loc[samp_idx, 'mass_1'])
                        else:
                            Mhe1.append(HeHe.loc[samp_idx, 'mass_1'])
                            Mhe2.append(HeHe.loc[samp_idx, 'mass_2'])
                    else:
                        porb.append(np.nan)
                        Mhe1.append(np.nan)
                        Mhe2.append(np.nan)
                df.loc[idxs_in_metbin, 'porb_HeHe'] = np.asarray(porb)
                df.loc[idxs_in_metbin, 'Mhe1_HeHe'] = np.asarray(Mhe1)
                df.loc[idxs_in_metbin, 'Mhe2_HeHe'] = np.asarray(Mhe2)

                # BH masses (based on which was born first)
                df.loc[idxs_in_metbin[~secondary_born_first], 'Mbh1'] = np.asarray(dco_form_sample[~secondary_born_first]['mass_1'])
                df.loc[idxs_in_metbin[secondary_born_first], 'Mbh1'] = np.asarray(dco_form_sample[secondary_born_first]['mass_2'])
                df.loc[idxs_in_metbin[~secondary_born_first], 'Mbh2'] = np.asarray(dco_form_sample[~secondary_born_first]['mass_2'])
                df.loc[idxs_in_metbin[secondary_born_first], 'Mbh2'] = np.asarray(dco_form_sample[secondary_born_first]['mass_1'])

                # get the spin tilts
                kick_info = pd.read_hdf(os.path.join(pop_path, Z, dat_file), key='kick_info')
                kick_info['bin_num'] = kick_info.index
                second_SN = kick_info.groupby('bin_num').last()
                second_SN_sample = second_SN.loc[idxs_to_sample]
                df.loc[idxs_in_metbin, 'SN_theta'] = np.asarray(second_SN_sample['delta_theta_total'])

                # get the evolutionary pathway (CE, doubleCE, SMT)
                evo_path_df = pd.DataFrame(index=list(bpp.index.unique()))
                CE_idxs = list(bpp.loc[((((bpp['kstar_1']>=13) & (bpp['kstar_2']<10)) | ((bpp['kstar_2']>=13) & (bpp['kstar_1']<10))) & (bpp['evol_type']==7))].index.unique())
                evo_path_df.loc[CE_idxs,'evo_pathway'] = 'CE'
                SMT_idxs = list(set(bpp.index.unique()) - set(bpp.loc[bpp['evol_type']==7].index.unique()))
                evo_path_df.loc[SMT_idxs,'evo_pathway'] = 'SMT'
                doubleCE_idxs = bpp.loc[(bpp['kstar_1']<10) & (bpp['kstar_2']<10) & (bpp['evol_type']==7)].index.unique()
                evo_path_df.loc[doubleCE_idxs,'evo_pathway'] = 'doubleCE'

                evo_path_sample = evo_path_df.loc[idxs_to_sample]
                df.loc[idxs_in_metbin, 'evo_pathway'] = np.asarray(evo_path_sample['evo_pathway'])

        # remove systems that merged after z=0, if specified
        if mergers_only==True:
            df = df.loc[df['tlb_merge'] > 0]

            # Make sure there are no NaN values left
            assert(all(df.isna().any())==False)

        # reorder columns
        if initC_only==False:
            if extra_info:
                df = df[['z_ZAMS','z_DCO','z_merge','tlb_ZAMS','tlb_DCO','tlb_merge','m1','m2','a','porb','e','Z','M1_ZAMS','M2_ZAMS','porb_ZAMS','e_ZAMS','Mbh1','Mbh2','secondary_born_first','Mbh1_birth','Mbh1_preMT','Mbh1_postMT','Mstar_preMT','Mstar_postMT','delta_t_MT','Mbh_HeBH','Mhe_HeBH','porb_HeBH','Mhe1_HeHe','Mhe2_HeHe','porb_HeHe','SN_theta','evo_pathway']]
            else:
                df = df[['z_ZAMS','z_DCO','z_merge','tlb_ZAMS','tlb_DCO','tlb_merge','m1','m2','a','porb','e','Z']]

        self.resampled_pop = df


    # --- Add in dimensionful units to each system in the resampled population [Mpc^-3 yr^-1] --- #
    # NOTE: this is currently only implemented for Illustris resampling method!!!
    def calculate_system_rates(self, method):
        # Currently, this only works for populations resampled with Illustris
        if method != 'illustris':
            raise ValueError("Adding dimensionful rate contributions for each system is currently only supported for the Illustris population model!")

        df = self.resampled_pop.copy()
        df['weight'] = np.zeros(len(df))
        # cycle through redshift bins
        for idx, (zlow,zhigh) in tqdm(enumerate(zip(self.redshift_bins[:-1],self.redshift_bins[1:])), total=len(self.redshift_bins)-1):
            # get systems that were born in this redshift bin
            df_cut = df.loc[(df['z_ZAMS'] >= zlow) & (df['z_ZAMS'] < zhigh)]
            if len(df_cut) == 0:
                continue
            # get metallicity CDF at this redshift
            met_cdf = self.met_cdf_at_redz[idx]
            # get total mass formed at all metallicities in this redshift bin [Msun Mpc^-3 (dt)^-1]
            Mform_at_z = self.sfr_pts[idx]   # Msun Mpc^-3 yr^-1
            # sort the unique Z values
            met_at_z = np.sort(np.unique(df_cut.Z))
            # evaluate metallicity grid in the CDF
            met_cdf_eval = met_cdf(np.log10(met_at_z))
            # take the midpoints between CDF evaluations and assign weights to each metallicity
            met_cdf_eval = met_cdf_eval[:-1] + (met_cdf_eval[1:]-met_cdf_eval[:-1])/2
            met_cdf_eval = np.concatenate([[0],met_cdf_eval,[1]])
            met_weights = met_cdf_eval[1:] - met_cdf_eval[:-1]
            # each unique metallicity at this redshift bin gets fractional star formation according to these weights
            fSFR_per_met = met_weights * Mform_at_z   # [Msun Mpc^-3 yr^-1]
            # find the CBC formation efficiency at each metallicity where there is a system
            for idx, met in enumerate(met_at_z):
                # get the mass turned into CBCs at this metallicity
                frac_mass = self.formation_efficiencies[met] * fSFR_per_met[idx]
                # split this fractional mass over all the systems at the metallicity
                Nsys = len(df_cut.loc[df_cut['Z'] == met])
                df_cut.loc[df_cut['Z'] == met, 'weight'] = frac_mass / Nsys

            # update the weights for these systems in the full dataframe
            df.update(df_cut)

        self.resampled_pop = df
