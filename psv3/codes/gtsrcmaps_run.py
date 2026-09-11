from gt_apps import srcMaps

base_dir = '/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs/'

outfile=base_dir + 'photon/srcmaps/psr_1G_1T_z105_seed_001_srcmpas.fits'
xmlfile = base_dir + '/photon/mock_psr_realizations/mock_4FGL_psr_seed_001.xml'
cmapfile = base_dir + 'lat_selected_1G_1T_z105_gtbin.fits'
ltcube_file = base_dir + 'photon/livetime_cubes/lat_selected_1G_1T_z105_ltcube.fits'
exposure_file = base_dir + 'photon/exposure_cubes/lat_selected_1G_1T_z105_expcube.fits'


# model = GtApp("gtmodel", "Likelihood")


srcMaps["srcmdl"] = str(xmlfile)
srcMaps["outfile"] = str(outfile)

srcMaps["irfs"] = "P8R3_SOURCE_V3"
srcMaps["evtype"] = 3
srcMaps["cmap"] = str(cmapfile)
srcMaps["expcube"] = str(ltcube_file)
srcMaps["bexpmap"] = str(exposure_file)



srcMaps["clobber"] = "yes"
########
# these are added due to run issues for gll_iem for energy bins starting from >=300M
########
# srcMaps["emapbnds"] = "no" # use this for the iem run; irrelevant for the ptsrc
srcMaps["chatter"] = 4 # little bit more log
# srcMaps["resample"] = "yes" # use this only for iem run
# srcMaps["rfactor"] = 1 # use this only for iem run

srcMaps.run()
