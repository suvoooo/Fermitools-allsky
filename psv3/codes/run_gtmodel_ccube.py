from gt_apps import model_map

base_dir = '/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs/'

srcmaps_file=base_dir + 'photon/srcmaps/psr_1G_1T_z105_seed_001_srcmaps.fits'
xmlfile = base_dir + 'photon/mock_psr_realizations/mock_4FGL_psr_seed_001.xml'
outfile = base_dir + 'photon/gtmodel/psr_1G_1T_z105_seed_001_gtmodel_ccube.fits'
ltcube_file = base_dir + 'photon/livetime_cubes/lat_selected_1G_1T_z105_ltcube.fits'
exposure_file = base_dir + 'photon/exposure_cubes/lat_selected_1G_1T_z105_expcube.fits'


# model = GtApp("gtmodel", "Likelihood")

model_map["srcmaps"] = str(srcmaps_file)
model_map["srcmdl"] = str(xmlfile)
model_map["outfile"] = str(outfile)

model_map["irfs"] = "P8R3_SOURCE_V3"
model_map["evtype"] = 3
model_map["expcube"] = str(ltcube_file)
model_map["bexpmap"] = str(exposure_file)

model_map["outtype"] = "ccube"
model_map['edisp_bins'] = 0 #-2

model_map["clobber"] = "yes"

model_map.run()
