# Flow of Steps to Prepare 16 Years of Fermi-LAT Mock Sky 

* A lot of it are just repitions of psv2 (10 years data generation); Check that [Readme](https://github.com/suvoooo/Fermitools-allsky/blob/master/fermitools_flow.md)

1. First steps are downloading the weekly photon and spacecraft files for a specified period of time. For us, this over 16 yrs of data.  
Corresponding codes are written in `generate_photon_urls.py` which generates relevant urls for the weekly photon and spacecraft files; Next is automated download via bash files `download_fermi_ph_weekly.sh` and `download_fermi_sc_weekly.sh`

2. Next step is combining the weekly spacecraft files into a single fits file; which is done using `merge_weekly_sc_astropy.py`
 - sadly the merged_spacecraft file seems to corrupted all the time, so just merge by hand!! 

3. Next step is to use gtselect to combine the photon files; Here essentially we select the energy range; 30 MeV to 1 TeV. We used `gtselect_comb1.py`


4. Next step is event selection and apply proper zenith angle cuts depending on the energy intervals; 

 - Here we are focusing on [incremental Fermi-Catalog paper](https://arxiv.org/abs/2201.11184) to get the corresponding zenith angle cuts ; 
 
 - In short: 30 MeV to 100 MeV: Zmax=80; 100 MeV to 300 MeV: Zmax=90; 300 MeV to 1 GeV: Zmax=100; 1 GeV to 1 TeV: Zmax=105

 - For this, we have used `gtselect_comb2_Zcut.py`
  
5. Moving onto selection of Good Time Intervals (GTI) via gtmktime; we have used `gtmktime_run.py`. Exact same conditions as used in psv2.   

6. Next step is bin the photons and we use gtbindef and gtbin. 
 - We decided on 6 bins/decade (log): For gtbindef we need ascii files with these bin boundaries, done using: `prepare_gtbindef_ascii.py`
 - then use the ascii files for gtbin. For gtbin irrespective of bin boundaries we always used order of the map as 11. 
 	- For healpix $N_{pix} = 12N_{side}^2$; Approx linear pixel scale: $\theta _{pix} \sim \sqrt{\frac{4\pi}{12 N_{side}^2}} = \frac{58.63^{\circ}}{N_{side}}$; 
 	- For pixel resolution close to $0.03^{\circ}, \, N_{side} \simeq \frac{58.63^{\circ}}{0.03}\simeq 1954$  
 	- Close to 1954 we have 2048; $log_2(2048) = 11$; Characteristic pixel size $\theta _{pix}\simeq \frac{58.63}{2048} \simeq 0.0286^{\circ}$.  
  - for gtbin we have inputs as below from an example run: 
  
  ```
  gtbin
  This is gtbin version HEAD
  Type of output file (CCUBE|CMAP|LC|PHA1|PHA2|HEALPIX) [HEALPIX]
  Event data file name[/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs/photon/lat_all_data_Zcuts_gti/lat_selected_300M_1G_z100_gti.fits] /d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs/photon/lat_all_data_Zcuts_gti/lat_selected_1G_1T_z105_gti.fits
  Output file name[lat_selected_300M_1G_z100_gtbin.fits] lat_selected_1G_1T_z105_gtbin.fits
  Spacecraft data file name[/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs/spacecraft/lat_spacecraft_weekly_merged_astropy.fits]
  Ordering Scheme (RING|NESTED) [RING]
  Order of the map (int between 0 and 12, included)[11]
  Coordinate system (CEL - celestial, GAL -galactic) (CEL|GAL) [GAL]
  Region, leave empty for all-sky[]
  Do you want Energy binning ?[yes]
  Algorithm for defining energy bins (FILE|LIN|LOG) [FILE]
  Name of the file containing the energy bin definition[gtbindef_300M_1G.fits] gtbindef_1G_1T.fits
  ```
 - One of the important checks for the binned map is to compare the counts in the gti file and corresponding binned file; should be very close; done via: `gtbin_fits_integ_check.py`
 
7. Next we move with gtltcube. Since gtltcube depends on gti files and we have 4 of them, gtltcube runs 4 times, following `gtltcube_run.py`. 

8. Move on to calculate binned exposure map with the gtexpcube2. Since it depends on both gtbin and livetime cube, ran it 4 times following `gtexpcube2_run.py` 
