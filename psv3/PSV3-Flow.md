# Flow of Steps to Prepare 16 Years of Fermi-LAT Mock Sky 

* A lot of it are just repitions of psv2 (10 years data generation); Check that [Readme](https://github.com/suvoooo/Fermitools-allsky/blob/master/fermitools_flow.md)

1. First steps are downloading the weekly photon and spacecraft files for a specified period of time. For us, this over 16 yrs of data.  
Corresponding codes are written in `generate_photon_urls.py` which generates relevant urls for the weekly photon and spacecraft files; Next is automated download via bash files download_fermi_ph_weekly.sh and download_fermi_sc_weekly.sh

2. Next step is combining the weekly spacecraft files into a single fits file; which is done using `merge_weekly_sc_astropy.py'
 - sadly the merged_spacecraft file seems to corrupted all the time, so just merge by hand!! 

3. Next step is to use gtselect to combine the photon files; Here essentially we select the energy range; 30 MeV to 1 TeV. We used `gtselect_comb1.py`


4. Next step is event selection and apply proper zenith angle cuts depending on the energy intervals; 

 - Here we are focusing on [incremental Fermi-Catalog paper](https://arxiv.org/abs/2201.11184) to get the corresponding zenith angle cuts ; 
 
 - In short: 30 MeV to 100 MeV: Zmax=80; 100 MeV to 300 MeV: Zmax=90; 300 MeV to 1 GeV: Zmax=100; 1 GeV to 1 TeV: Zmax=105

 - For this, we have used `gtselect_comb2_Zcut.py`
  
5. Moving onto selection of Good Time Intervals (GTI) via gtmktime; we have used `gtmktime_run.py`. Exact same conditions as used in psv2.   
