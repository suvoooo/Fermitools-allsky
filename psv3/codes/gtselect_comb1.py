'''
this is the first gtsselect run;
all energy i.e. 30 MeV to 1 TeV and all photon 
no event selection cuts yet!
'''

import os
from gt_apps import filter as gtselect

# 1. Define the parameters for the all-sky combination
gtselect['evclass'] = 'INDEF'
gtselect['evtype'] = 'INDEF'
gtselect['infile'] = '@all_photon_files.txt'     # list of downloaded FT1 files
gtselect['outfile'] = 'lat_alldata_16yrs.fits'  # combined output file
gtselect['ra'] = 0
gtselect['dec'] = 0
gtselect['rad'] = 180                     # Full sky
gtselect['tmin'] = 'INDEF'
gtselect['tmax'] = 'INDEF'
gtselect['emin'] = 30                     # 30 MeV
gtselect['emax'] = 1000000                # 1 TeV
gtselect['zmax'] = 180                    # No zenith cut yet

# 2. Run the tool
print("Starting gtselect to combine all-sky files...")
gtselect.run()
print("Combination complete. Output saved to lat_alldata.fits")
