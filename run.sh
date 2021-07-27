#!/bin/bash/

pop='/Users/michaelzevin/research/cosmic/populations/v3.4/bbh_merged_qc4'
method=illustris

python resample \
--Nresamp 1000000 \
--population ${pop} \
--method ${method} \
--output-path './tests/resampled_illustris_pessimistic.hdf5' \
--zmin 0 \
--zmax 20 \
--Zmin 0.0001 \
--Zmax 0.04 \
--sigmaZ 0.5 \
--verbose \
--filter pessimistic_CE \
#--mergers-only \

