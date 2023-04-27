#!/bin/bash/

pop='/Users/michaelzevin/research/cosmic/populations/acc_efficiency/alpha1_qc4_eddfac1e0/acclim_0'
method=illustris

python resample \
--Nresamp 100000 \
--population ${pop} \
--method ${method} \
--output-path '/Users/michaelzevin/research/cosmic/populations/acc_efficiency/alpha1_qc4_eddfac1e0/acclim_0/resampled_illustris.hdf5' \
--zmin 0 \
--zmax 20 \
--Zmin 0.0001 \
--Zmax 0.04 \
--sigmaZ 0.5 \
--verbose \
--extra-info \
--mergers-only \
#--filter bbh \

