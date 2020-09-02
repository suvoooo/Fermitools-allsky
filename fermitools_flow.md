## Flow of work procedure in generating data using gtmodel... 

Download the full set of weekly photon files. 

Download the mission spacecraft file. 

For these two steps follow direction [here](https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/LAT_weekly_allsky.html).

For steps till gtexpcube the above link will be useful. 

Combine the weekly files into a single file without removing any events. This can be done using gtselect tool. (Takes whole lot of time)

Now we filter the data for the proper event class and remove earth limb contamination. 
	[Recommended](https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_DP.html#PhotonClassification) use for point source and moderately extended sources and also for galactic [diffuse analysis](https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data_Exploration/Data_preparation.html): evclass - 128 (P8R3_SOURCE_V2), include evtype - 3 cut to use:
    
	both front and back converting event types. 
	remove contamination by limiting reconstructed zenith angle of 90 degrees or less. 

The earth limb lies at a zenith angle of 113 degrees, so suggested value of 90 degrees provides protection against significant contamination by atmospheric gammas. Also improve the PSF by excluding events with reconstructed energies below 1GeV.      



Next, correct the exposure for the events that we have filtered out. Use gtmktime tool. 
One issue here --- zenith cut was used in the previous gtselect run but we won't correct for it here. As mentioned in docs 'for an all sky analysis an ROI based zenith cut would eliminate the entire data-set'. We will refer back to this later when we use gtltcube tool.   

#######################
### gtmktime
#######################

gtmktime creates good time intervals (GTIs) based on selection made using the spacecraft data file variables. In a more general term it updates the GTIs extension and make cuts based on spacecraft parameters contained in the livetime and pointing history FITS file. GTI is a time range when the data can be considered valid.   


time selection recommendation is given [here](https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data_Exploration/Data_preparation.html). 

for gtmktime within Filter expression we select (DATA_QUAL>0) && (LAT_CONFIG==1) && (IN_SAA !=T)
	what this does is -- 
    
    1. exclude time periods when some spacecraft event has affected the quality of the data, 
    2. ensures the lat instrument was in normal science data-taking mode, 
    3. select times when the spacecraft is not in the Southern Atlantic Anomaly (SAA).    	  

----------------------------------------------------


After gtmktime, we bin the data for exposure correction. Here we need gtbin and gtbindef. 


#############################
### gtbin and gtbindef
#############################

gtbin will be mostly used to produce count maps.  

for gtbin -- the binning needs to be defined and this can also created using another fermitool gtbindef. Here we used it for producing HEALPIX type file as output. 
Relevant parameters are Ordering scheme Ring, Order of map 8, coordinate system GAL, and finally the file produced by gtbindef can be passed to specify 
particular bin boundaries.  

for gtbindef the binning needs to be user defined, for Fermilat analysis the unit is in MeV. See docs.

for gtbindef we need to create an ascii file with energy bins specified (with MeV units). ex: 
30	50
50	90
90	...etc..

for this work we use the Fermi-energy binning as mentioned in the original 4FGL paper in Table 2 as a start exercise.  

----------------------------------------------------




Next we compute the livetime cube using gtltcube. This is actually a precursor step for exposure correction. 
Once again we refer to the zenith angle cut made with gtselect earlier, to need to correct the livetime. Use 'zmax' option on the command and match the values 
used in gtselect. This modifies the livetime calculation to account for the events we removed earlier.  

##############################
###  gtltcube (set zmax=90)
##############################

compute livetime as a function of sky position and off-axis angle. gtltcube creates a livetime cube which is a Healpix table, covering the entire sky, of the integrated livetime
as a function of inclination wrt LAT z axis. Off-axis angle: Angle between the direction to a source and the instrument z-axis, LAT irfs depend on the off-axis angle.  

Livetime: accumulated time during which the LAT is actively taking event data. 

parameters passed on: 

    event file : file created with gtmktime tool after applying necessary selection criterion. 
    spacecraft file: spacecraft data extension file, downloaded from Fermi-lat server, 
    output file: specify a new file name.
    dcostheta (step size in cos): inclination angle binning represented as the cosine of the off-axis angle. used value: 0.09. 
    binsz (pixel size in degrees): size of the spatial grid in degrees. 

this takes time (depends on the angular bin size), once the livetime cube is generated now we are ready to compute the exposure map.  

----------------------------------------------------



Before the next step some important points about Fermi-Lat IRFs. IRF is factored in 3 terms, resolution (PSF), effective area of detector & energy dispersion. Each event class and event type has its own IRFs (these were previously selected using gtlike, so check consistency). Pass 8 data release (P8R3) defines 3 event types partitions: resolution (4 types of PSFs), effective area (Front/Back), Energy Dispersion (4 types of EDISPs).  

In the documentation for an example analysis with 3c279 blazar the information below are given.  
The large PSF of LAT implies at low energies sources from well outside the counts cube could affect the source we are analyzing (maybe less important for all sky data ?). To compensate for this we must create an exposure map that that include sources 10 degrees beyond the ROI. The ROI in the given example already have 15 degrees radius.   

The exposure needs to be recalculated if ROI, zenith angle, time, event class or energy selections applied to the data are changed. For binned analysis this also includes the spatial and energy binning of the 3D counts map. The scripting for generating exposure map of the entire sky is way simpler than calculation of exposure for a particular sky. This also explains the fact that selection of ROI is way simpler.   

#################
### gtexpcube2
################# 

generates (binned) exposure map, or a set of exposure maps for different energies, from a livetime cube written by gtltcube.
parameters:	

        livetime cube file: file created with gtltcube. the filename that was provided for counts cube-- it will use the information from that file to define the geometry of the exposure map.
        
		counts map file: file created with gtbin (binned healpix map)
        
		response function: P8R3_SOURCE_V2 (default for evclass = 128 and evtype = 3). recommended class for most analyses, provides good sensitivity for analysis of point sources and moderately extended sources. P8R3_CLEAN_V2 (evclass = 256, evtype = ?) and ULTRACLEAN and more all have backrground lower background rate than the source class (P8R3_SOURCE_*).  	 
 

----------------------------------------------------


#######################
###  gtsrcmaps
#######################
Convolves source model components with the instrument response. 

gtsrcmaps creates model count maps for use with the binned likelihood analysis. The source xml file is necessary, which is 4FGL xml file (available as 8 yr source catalog in xml format). 


parameters:	

        Exposure hypercube file: file created with gtltcube 
		Counts map file: Healpix file created with gtbin. 
		source model file: mock catalogue .xml file
		binned exposure map: file created with gtexpcube2. 
		Response function: same as used in gtexpcube2 :P8R3_SOURCE_V2, Check [here](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html) for more info. 


----------------------------------------------------


##################
### gtmodel 
##################

Make a model skymap using gtmodel. This is our final step. It is suggested in the documentation to use gtlike to create the model map of the region based 
on the fit parameters. Here we didn't use gtlike. Because rather than a single source and a particular ROI, we are interested in all sky map. 

parameters : 	

        source maps (or counts map) file: created with gtsrcmaps. (for using counts map to create model image check [here](https://fermi-hero.readthedocs.io/en/latest/galactic_center/science_tool_images.html).)
		source model file: same as used in gtsrcmaps (.xml file catalog)
		output file: write a convenient name .fits
		Response functions: consistent with previous one 
		Exposure cube: file created with gtltcube
		binned exposure map: file created with gtexpcube file. 
		
		 
splits out some warning but, no worries :>). 

----------------------------------------------------


!!!! done !!!!