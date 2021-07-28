import numpy as np
import pandas as pd

import os
import h5py
import pdb
from tqdm import tqdm

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
    def __init__(self, method, sigmaZ, zmin=None, zmax=None, Zmin=None, Zmax=None):
        # Instantiates the SFH class that will be used to draw redshifts and metallicities
        self.method = method
        self.zmin = zmin
        self.zmax = zmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        if method is not 'illustris':
            self.sigmaZ = sigmaZ

        ### special treament for reading in Illustris data
        if method=='illustris':
            with h5py.File('./data/TNG100_L75n1820TNG__x-t-log_y-Z-log.hdf5', 'r') as f:
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

            # slice arrays based on Zmin and Zmax, if specified
            if (Zmin is not None) and (Zmin > met_bins.min()):
                # we only care about stuff above some threshold metallicity
                metal_enough_idx = np.argwhere(met_bins >= Zmin)[0][0]
                met_bins = np.squeeze(met_bins[metal_enough_idx:])
                # add lowest metallicity to beginning of met bins
                met_bins = np.insert(met_bins, 0, Zmin)
                Mform = np.squeeze(Mform[:, (metal_enough_idx-1):])
            if (Zmax is not None) and (Zmax < met_bins.max()):
                # and remove things that have metallicities above ~2Zsun
                too_metal_idx = np.argwhere(met_bins > Zmax)[0][0]
                met_bins = np.squeeze(met_bins[:too_metal_idx])
                # add highest metallicity to end of array
                met_bins = np.append(met_bins, Zmax)
                Mform = np.squeeze(Mform[:, :too_metal_idx])

            # Make sure everything has the correct dimensions
            assert (Mform.shape[0]==len(time_bins)-1) and (Mform.shape[0]==len(redshift_bins)-1) and (Mform.shape[1]==len(met_bins)-1), "Dimensions of bin heirhgts in Mform ({:d,:d}) do not match the time ({:d}), redshift ({:d}), and metallicity ({:d}) bins!".format(Mform.shape[0], Mform.shape[1], len(time_bins), len(redshift_bins), len(met_bins))

            # get times and metallicity bin centers (in logspace)
            times = 10**(np.log10(time_bins[:-1])+((np.log10(time_bins[1:])-np.log10(time_bins[:-1]))/2))
            mets = 10**(np.log10(met_bins[:-1])+((np.log10(met_bins[1:])-np.log10(met_bins[:-1]))/2))

            # calculate redshifts for the log-spaced centers (for interpolation)
            redshifts = []
            for t in tqdm(times):
                redshifts.append(z_at_value(cosmo.age, t*u.yr))
            redshifts = np.asarray(redshifts)

            # time duration in each bin
            dt = time_bins[1:] - time_bins[:-1]

            self.Mform = Mform
            self.times = times
            self.mets = mets
            self.redshifts = redshifts
            self.dt = dt
            self.time_bins = time_bins
            self.met_bins = met_bins
            self.redshift_bins = redshift_bins


    # --- Methods for drawing redshifts and metallcities --- #
    def draw_redshifts(self, Nsamp, sfr_mdl='2017'):
        """
        Draws redshifts based on chosen SFH
        """

        if self.method=='illustris':
            # reverse so that z=0 is at beginning of array
            redshift_grid = self.redshifts[::-1]
            sfr_pts = np.sum(self.Mform[::-1], axis=1) / self.dt[::-1] / 100**3   # simulation from 100 Mpc^3 bin
            sfr_pts = sfr_pts
        else:
            redshift_grid = np.linspace(self.zmin, self.zmax, 10000)
            sfr_pts = sfr_z(redshift_grid, mdl=sfr_mdl)

        # create inverse cdf interpolant
        sfr_interp = interp1d(redshift_grid, sfr_pts)
        sfr_cdf = cumulative_trapezoid(sfr_pts, redshift_grid, initial=0)
        sfr_cdf /= sfr_cdf.max()
        sfr_icdf_interp = interp1d(sfr_cdf, redshift_grid, fill_value="extrapolate")

        # draw from icdf
        redshift_draws = sfr_icdf_interp(np.random.uniform(0,1, size=Nsamp))
        self.redshift_draws = redshift_draws


    def draw_metallicities(self, Nsamp):
        """
        Draws metallicities with cdfs constructed in log-space for metallicity
        """

        if self.method=='illustris':
            # Precompute metallicity inverse CDFs for each redshift bin
            met_icdf_interps = []
            for idx, metdist_at_z in enumerate(self.Mform):
                if np.sum(metdist_at_z) != 0.0:
                    met_cdf = cumulative_trapezoid(metdist_at_z, np.log10(self.mets), initial=0)
                    met_cdf /= met_cdf.max()
                    met_icdf_interps.append(interp1d(met_cdf, self.mets, fill_value="extrapolate"))
                else:
                    # no SF at this time in any metallicity bin
                    met_icdf_interps.append(np.nan)

            redshift_indices = np.searchsorted(self.redshift_bins[::-1], self.redshift_draws)   # has to be ascending for searchsorted, so we're going from z=0 to z=zmax
            redshift_indices -= 1   # convert from bins to indices needed for Mbin
            met_icdf_interps = met_icdf_interps[::-1]   # rearrange these from z=0 to z=zmax as well

            met_draws = []
            for idx, redz_idx in enumerate(tqdm(redshift_indices)):
                met_icdf_interp = met_icdf_interps[redz_idx]
                try:
                    met_draws.append(met_icdf_interp(np.random.random()))
                except:
                    print('Failed at z={:0.3f}'.format(self.redshift_draws[redz_idx]))
                    met_draws.append(self.Zmin)     # Shouldn't happen, but just to catch it

            met_draws = np.asarray(met_draws).flatten()

        elif self.method=='truncnorm':
            log_met_draws = []
            for idx, redz in enumerate(tqdm(self.redshift_draws)):
                Zdist = metal_disp_truncnorm(redz, self.sigmaZ, self.Zmin, self.Zmax)
                log_met_draws.append(Zdist.rvs())

            met_draws = 10**np.asarray(log_met_draws)

        elif self.method=='corrected_truncnorm':
            truncated_mean_to_gaussian_mean = corrected_means_for_truncated_lognormal(self.sigmaZ, self.Zmin, self.Zmax)
            log_met_draws = []
            for idx, redz in enumerate(tqdm(self.redshift_draws)):
                Zdist = metal_disp_truncnorm_corrected(redz, truncated_mean_to_gaussian_mean, self.sigmaZ, self.Zmin, self.Zmax)
                log_met_draws.append(Zdist.rvs())

            met_draws = 10**np.asarray(log_met_draws)

        self.met_draws = met_draws


    # --- Resampling method --- #
    def resample(self, pop_path, pop_filters=None, mergers_only=False):
        """
        Resamples populations living at `pop_path` based on the drawn formation redshifts and metallicities
        """

        # create dataframe for housing the resampled data
        df = pd.DataFrame(np.asarray([self.redshift_draws, self.met_draws]).T, columns=['z_ZAMS','Z_draw'])
        df['tlb_ZAMS'] = cosmo.lookback_time(np.asarray(df['z_ZAMS'])).to(u.Myr).value

        pop_mets = np.sort([float(x) for x in os.listdir(pop_path) if '0.' in x])
        pop_mets_str = sorted([x for x in os.listdir(pop_path) if '0.' in x])

        # find closest metallicity run (in log-space) to the met_draw
        met_samp = [pop_mets[np.argmin(np.abs(np.log10(pop_mets)-logZ))] for logZ in np.log10(self.met_draws)]
        df['Z_samp'] = np.asarray(met_samp)

        # now, read in bpp arrays and get the DCO formation parameters (tlb_DCO, z_DCO, t_insp, tlb_merge, z_merge, m1, m2, a, porb, e)
        df['tlb_DCO'] = np.nan
        df['z_DCO'] = np.nan
        df['t_delay'] = np.nan
        df['t_insp'] = np.nan
        df['tlb_merge'] = np.nan
        df['z_merge'] = np.nan
        df['m1'] = np.nan
        df['m2'] = np.nan
        df['a'] = np.nan
        df['porb'] = np.nan
        df['e'] = np.nan

        # get lookback time to redshift interpolant
        z_grid = np.logspace(-5, 2, 100000)
        z_grid = np.insert(z_grid, 0, 0)
        tlb_grid = cosmo.lookback_time(z_grid).to(u.Myr).value
        tlb_to_z_interp = interp1d(tlb_grid, z_grid)

        for idx, Z in tqdm(enumerate(pop_mets_str), total=len(pop_mets_str)):
            dat_file = [x for x in os.listdir(os.path.join(pop_path, Z)) if 'dat' in x][0]
            bpp = pd.read_hdf(os.path.join(pop_path, Z, dat_file), key='bpp')

            # apply filters to bpp array, if specified
            if pop_filters is not None:
                for filt in pop_filters:
                    if filt in filters._filters_dict.keys():
                        bpp = filters._filters_dict[filt](bpp)
                    else:
                        raise ValueError('The filter you specified ({:s}) is not defined in the filters function!'.format(filt))

            # get point of DCO formation
            dco_form = bpp.loc[((bpp['kstar_1']==13)|(bpp['kstar_1']==14)) \
                               & ((bpp['kstar_2']==13)|(bpp['kstar_2']==14))].groupby('bin_num').first()
            dco_merge = bpp.loc[((bpp['kstar_1']==13)|(bpp['kstar_1']==14)) \
                               & ((bpp['kstar_2']==13)|(bpp['kstar_2']==14))].groupby('bin_num').last()

            # if there are no systems in the population model that satisfy the  criteria, just leave all their entries and NaNs
            if (len(dco_form)==0) or (len(dco_merge)==0):
                print("No matching systems found in the population model for Z={:s}!".format(Z))
                continue

            # random choose systems from this model
            idxs_in_metbin = list(df.loc[df['Z_samp']==pop_mets[idx]].index)
            idxs_to_sample = np.random.choice(list(dco_form.index), size=len(idxs_in_metbin), replace=True)
            dco_form_sample = dco_form.loc[idxs_to_sample]
            dco_merge_sample = dco_merge.loc[idxs_to_sample]

            # set these values in the dataframe
            df.loc[idxs_in_metbin, 'tlb_DCO'] = np.asarray(df.loc[idxs_in_metbin, 'tlb_ZAMS']) - np.asarray(dco_form_sample['tphys'])
            df_tmp = df.loc[idxs_in_metbin]   # Needed to get indices for things that merge after z=0
            idxs_in_metbin_tH = list(df_tmp.loc[df_tmp['tlb_DCO'] > 0].index)
            df.loc[idxs_in_metbin_tH, 'z_DCO'] = tlb_to_z_interp(np.asarray(df.loc[idxs_in_metbin_tH, 'tlb_DCO']))

            df.loc[idxs_in_metbin, 't_delay'] = np.asarray(dco_merge_sample['tphys'])
            df.loc[idxs_in_metbin, 't_insp'] = np.asarray(dco_merge_sample['tphys']) - np.asarray(dco_form_sample['tphys'])
            df.loc[idxs_in_metbin, 'tlb_merge'] = np.asarray(df.loc[idxs_in_metbin, 'tlb_ZAMS']) - np.asarray(dco_merge_sample['tphys'])
            df_tmp = df.loc[idxs_in_metbin]   # Needed to get indices for things that merge after z=0
            idxs_in_metbin_tH = list(df_tmp.loc[df_tmp['tlb_merge'] > 0].index)
            df.loc[idxs_in_metbin_tH, 'z_merge'] = tlb_to_z_interp(np.asarray(df.loc[idxs_in_metbin_tH, 'tlb_merge']))

            df.loc[idxs_in_metbin, 'm1'] = np.maximum(np.asarray(dco_form_sample['mass_1']), np.asarray(dco_form_sample['mass_2']))
            df.loc[idxs_in_metbin, 'm2'] = np.minimum(np.asarray(dco_form_sample['mass_1']), np.asarray(dco_form_sample['mass_2']))
            df.loc[idxs_in_metbin, 'a'] = np.asarray(dco_form_sample['sep'])
            df.loc[idxs_in_metbin, 'porb'] = np.asarray(dco_form_sample['porb'])
            df.loc[idxs_in_metbin, 'e'] = np.asarray(dco_form_sample['ecc'])

        # remove systems that merged after z=0, if specified
        if mergers_only==True:
            df = df.loc[df['tlb_merge'] > 0]

            # Make sure there are no NaN values left
            assert(all(df.isna().any())==False)

        # reorder columns
        df = df[['z_ZAMS','z_DCO','z_merge','tlb_ZAMS','tlb_DCO','tlb_merge','m1','m2','a','porb','e','Z_draw','Z_samp']]

        self.resampled_pop = df



    # --- Save to disk --- #
    def save(self, output_path):
        """
        Resamples populations living at `pop_path` based on the drawn formation redshifts and metallicities
        """
        self.resampled_pop.to_hdf(output_path, key='underlying')
