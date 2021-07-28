#!/bin/bash/

pop='/Users/michaelzevin/research/cosmic/populations/gw190814/sana_optimistic_delayed_alpha1_kick0'
method=truncnorm

python resample \
--Nresamp 100000 \
--population ${pop} \
--method ${method} \
--output-path '/Users/michaelzevin/research/resample_population/tests/resampled_gw190814_truncnorm.hdf5' \
--zmin 0 \
--zmax 20 \
--Zmin 0.0001 \
--Zmax 0.04 \
--sigmaZ 0.5 \
--verbose \
--filter bbh \
#--mergers-only \

