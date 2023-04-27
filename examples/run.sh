#!/bin/bash/

poppath='/Users/michaelzevin/research/posydon/population_models/test_1M.hdf5'
cosmopath='/Users/michaelzevin/research/git/resample-population/data/TNG100_L75n1820TNG__x-t-log_y-Z-log.hdf5'
outputpath='/Users/michaelzevin/research/posydon/population_models/test_1M_resampled_illustris.hdf5'
method=illustris

python ../resample \
--Nresamp 100000 \
--population ${poppath} \
--method ${method} \
--cosmo-path ${cosmopath} \
--output-path ${outputpath} \
--zmin 0.001 \
--zmax 20 \
--sigmaZ 0.5 \
--mergers-only \
--verbose \

