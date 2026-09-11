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

9. Next as usual is srcmaps: Run using `gtsrcmaps_run.py' . Check the header if they are correctly produced: `check_agn_psr_srcmaps_more.py'

10. Finally, use gtmodel: Run using `run_gtmodel_ccube.py' . Check the header if they are correctly produced: `gtmodel_check.py'


---------------------------------------------------------


#### Decision on Patch Size and Number of Patches; 

* Given the pixel size in Healpix $\sim 0.03^{\circ}$, we wanted to have reasonably large but still managable by neural-net relatively easy (computationally); Decided on patch size to be $512\times 512$ pixels i.e. covering about $14.6^{\circ}\times 14.6^{\circ} \approx 214 \, \text{deg}^2$. 

	- Number of sources/patch: Given the area of the sky $4\pi$ sr, basically about $4\pi \left(\frac{180}{\pi}\right)^2 \approx 41253\, \text{deg}^2$. Total number of sources (16yrs cat) in LAT: 7224; so number density of sources $\rho = \frac{7224}{41253}\approx 0.175 \, \text{deg}^{-2}$. So, roughly 1 source/$5.7\, \text{deg}^2$. So expected number of sources given area $214 \, \text{deg}^2$ is about 37. (quite a lot!)
	
* Now about how many patches to effectively cover the sky, to have minimal overlap and avoid edge effect (i.e. try to keep sources away from the edges of the patches): Turned out 432 gives the best of them all; with this layout, it covers the full sky in our sampling test, with each sky position appearing in 2.24 patches on average. Approximately $98.9\%$ of the sky lies at least $1.15^{\circ}$ from all edges of at least one patch, allowing a source’s bounding box to fit fully within that patch. The remaining $1.1\%$ is covered only near patch edges.	  

| Num. Patches  | Av. Coverage/sky point  | Sky never $\geq 1.15^{\circ}$ inside any patch  | 
|---|---|---|
| 192   | 1.00x  | 30.8%  |
| 300  | 1.57x  | 11.0%  |
| _432_  | 2.24x  | 1.1%  |
| 588 | 3.07x | 0.0% |
| 768 | 4.00x | 0.0% |


----------------------------------- 
