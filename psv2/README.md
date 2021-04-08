# Point Source Generation V2 
### Exploit Full Detector Potential, Added Source Classes, Added Time Variability of Blazars  

The notebook is quite self-explanatory for generating mock xml files.   

For the `fermitools` part we follow the steps as described [here](https://github.com/suvoooo/Fermitools-allsky/blob/master/fermitools_flow.md).   

The detailed steps however are somewhat different.  

* Treat FRONT and BACK events differently. 
* `evtype= 1,2,3` represents FRONT, BACK and combined events respectively
* Energy and Spatial binning is also different from PSv1. 
* 300 MeV — 500 MeV (Front and Back Events:  Same Spatial Resolution 0.8 degree)
* 500 MeV – 1 GeV (Front and Back:  0.8) 
* 1 GeV – 2 GeV (Front:  0.4, Back:  0.8) ;  Healpix Nside = 7, 8, 9 corresponds to 0.8, 0.4, 0.2 degree (used in `gtbin` tool).  
* 2 GeV – 7 GeV (Front:  0.2, Back:  0.4) 
* 7 GeV – 20 GeV (Front:  0.1, Back:  0.2)  
* 20 GeV – 1 TeV (Front:  0.1, Back:  0.2◦)
* For Zenith angle cut (in `gtselect` tool) For energy>1 GeV: Zenith Angle Cut 105 degree for both Front and Back type events. For energy<1 GeV: 90 degree cut for Back Events, 100 degree for Front Events. 

Apart  from  the `evtype` keyword  used  for  event  selection  (FRONT  and  BACK) in `gtselect`, `gtexpcube2`, `gtsrcmaps` we also add `edisp_bins=-2` in `gtexpcube2` and in `gtmodel`. 

# BLLac Example Sky-Map for 1-2 GeV Bin 

![BLLac 1-2mock](https://github.com/suvoooo/Fermitools-allsky/blob/master/psv2/bll1_2mock.jpg)

# PSR Example Sky-Map for 1-2 GeV Bin 

![PSR 1-2mock](https://github.com/suvoooo/Fermitools-allsky/blob/master/psv2/psr1_2mock.jpg)
