import numpy as np
import pandas as pd

def pessimistic_CE(bpp, initC, pessimistic_merge=[0,1,2,7,8,10,11,12]):
    CE = bpp.loc[bpp['evol_type']==7]
    # check which one is the donor
    CE_star1donor =  CE.loc[CE['RRLO_1']>=CE['RRLO_2']]
    CE_star2donor =  CE.loc[CE['RRLO_1']<CE['RRLO_2']]
    # Get the ones that would merge
    CE_merge_star1donor = CE_star1donor.loc[CE_star1donor['kstar_1'].isin(pessimistic_merge)].index.unique()
    CE_merge_star2donor = CE_star2donor.loc[CE_star2donor['kstar_2'].isin(pessimistic_merge)].index.unique()
    # drop these
    bpp_cut = bpp.drop(np.union1d(CE_merge_star1donor,CE_merge_star2donor))
    initC_cut = initC.drop(np.union1d(CE_merge_star1donor,CE_merge_star2donor))
    return bpp_cut, initC_cut

def bbh_filter(bpp, initC):
    bbh_idxs = bpp.loc[(bpp['kstar_1']==14) & (bpp['kstar_2']==14)].index.unique()
    bpp_cut = bpp.loc[bbh_idxs]
    initC_cut = initC.loc[bbh_idxs]
    return bpp_cut, initC_cut

def nsbh_filter(bpp, initC):
    nsbh_idxs = bpp.loc[((bpp['kstar_1']==13) & (bpp['kstar_2']==14)) |\
         ((bpp['kstar_1']==14) & (bpp['kstar_2']==13))].index.unique()
    bpp_cut = bpp.loc[nsbh_idxs]
    initC_cut = initC.loc[nsbh_idxs]
    return bpp_cut, initC_cut

def bns_filter(bpp, initC):
    bns_idxs = bpp.loc[(bpp['kstar_1']==13) & (bpp['kstar_2']==13)].index.unique()
    bpp_cut = bpp.loc[bns_idxs]
    initC_cut = initC.loc[bns_idxs]
    return bpp_cut, initC_cut

def HMXB_filter(bpp, initC):
    FirstCOForm = bpp.loc[(bpp['kstar_1']==13) | (bpp['kstar_1']==14)].groupby('bin_num').first()
    HMXBs = FirstCOForm.loc[(FirstCOForm['mass_2'] >= 5) & (FirstCOForm['sep'] > 0)]
    bpp_cut = bpp.loc[HMXBs.index]
    initC_cut = initC.loc[HMXBs.index]
    return bpp_cut, initC_cut

_filters_dict = {'pessimistic_CE': pessimistic_CE, \
                  'bbh': bbh_filter, \
                  'nsbh': nsbh_filter, \
                  'bns': bns_filter, \
                  'hmxb': HMXB_filter}
