import os

# Directories 

name_ims = "fpm_128"
name_obs = "NACO_M"
rep_bulk_ims = os.environ.get('SCRATCH')+'/'+name_ims
print(rep_bulk_ims)
rep_bulk_obs = os.environ.get('SCRATCH')+'/'+name_ims+'_'+name_obs
print(rep_bulk_obs)
rep_output = './output/'+name_ims+'_'+name_obs
