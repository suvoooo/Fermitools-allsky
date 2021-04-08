import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

gll_psc_v27 = fits.open('./PATH/My Path/gll_psc_v27.fit')
gll_psc_v27.info()

gll_psc_v27_list1 = gll_psc_v27[1]
print ('data shape: ', gll_psc_v27_list1.data.shape)

print ('only column names list:', '\n' , gll_psc_v27_list1.columns.names)

gll_psc_v27_list1_data = gll_psc_v27_list1.data

## check the column names before 
v23_Name = gll_psc_v27_list1_data['Source_Name']
v23_RA = gll_psc_v27_list1_data['RAJ2000']
v23_DEJ = gll_psc_v27_list1_data['DEJ2000']
v23_GLON = gll_psc_v27_list1_data['GLON']
v23_GLAT = gll_psc_v27_list1_data['GLAT']
v23_PEn = gll_psc_v27_list1_data['Pivot_Energy']
v23_F1000 = gll_psc_v27_list1_data['Flux1000']
v23_EnF1000 = gll_psc_v27_list1_data['Energy_Flux100']
v23_spectype = gll_psc_v27_list1_data['SpectrumType']
v23_PLIndex = gll_psc_v27_list1_data['PL_Index']
v23_PLFD = gll_psc_v27_list1_data['PL_Flux_Density']   
v23_FDensity = gll_psc_v27_list1_data['LP_Flux_Density']
v23_Index = gll_psc_v27_list1_data['LP_Index']
v23_Beta = gll_psc_v27_list1_data['LP_beta']
v23_Scurve = gll_psc_v27_list1_data['LP_SigCurv']
v23_Signif_Avg = gll_psc_v27_list1_data['Signif_Avg']
v23_Npred = gll_psc_v27_list1_data['Npred']
v23_Class1 = gll_psc_v27_list1_data['CLASS1']
v23_VarIndex = gll_psc_v27_list1_data['Variability_Index']
v23_FracVar = gll_psc_v27_list1_data['Frac_Variability']
v23_FluxHist = gll_psc_v27_list1_data['Flux_History']
v23_PLExpfactor = gll_psc_v27_list1_data['PLEC_Expfactor']
v23_PLECIndex = gll_psc_v27_list1_data['PLEC_Index']
v23_PLECExpIndex = gll_psc_v27_list1_data['PLEC_Exp_Index']
v23_PLECFD = gll_psc_v27_list1_data['PL_Flux_Density']
v23_PLECScurve = gll_psc_v27_list1_data['PLEC_SigCurv']

v23_df = pd.DataFrame(data=v23_Name, columns=['Name'])

### select relevant columns for AGN catalog generation

list_of_cols_v23 = [v23_RA, v23_DEJ, v23_PEn, v23_spectype, v23_FDensity, v23_Index, v23_Beta,  
                    v23_PLECIndex, v23_PLECFD, v23_PLECExpIndex, v23_PLExpfactor, v23_PLIndex, v23_PLFD, v23_Class1]

list_of_cols_strings_v23 = ['RA', 'DEC', 'Piv_En', 'Spec_Type', 'LP_FD', 'LP_index', 'LP_beta', 
                            'PLEC_Index', 'PLEC_FD', 'PLEC_Index2', 'PLEC_Expf', 'PL_Index', 'PL_FD', 'Class1']

for y in range(len(list_of_cols_v23)):
    v23_df[list_of_cols_strings_v23[y]] = list_of_cols_v23[y]

# v23LP_df.head(3)
print ('check selected data frame shape: ', v23_df.shape)

# first check the unique elements of column Class1
Class1_unique = v23_df.Class1.unique()
print ('check unique elements in Class1: ', Class1_unique)

Spec_type_unique = v23_df.Spec_Type.unique()
print ('check unique elements in Spec type: ', Spec_type_unique)

v23_df.dropna(inplace=True)
print ('check shape after dropping nan rows: ', v23_df.shape)

### select sub-data frame based on source tag

selected_class = ['bcu  ', 'bll  ', 'fsrq ', 'spp  ', 'PSR  ', 'FSRQ ', 'snr  ', 'BLL  ', 'SNR  ', 
                  'psr  ', 'PWN  ', 'pwn  ', 'BCU  ']


v23_df_selected = v23_df[v23_df['Class1'].isin(selected_class)]

### checks 

print ('Data frame shape: ', v23_df.shape)
print ('Only AGN Dataframe shape: ', v23_df_selected.shape)
print ('cross check the unique elements in Class column and length: ', v23_df_selected.Class1.unique(), 
      len(v23_df_selected.Class1.unique()))

#####################
# convert the dataframe columns to list
#####################

RA_list = v23_df_selected['RA'].tolist()
# print ('check ra elements', RA_list[1], RA_list[2])
DEC_list = v23_df_selected['DEC'].tolist()

Piv_Energy = v23_df_selected['Piv_En'].tolist()

PL_Index = v23_df_selected['PL_Index'].tolist()

PL_FD = v23_df_selected['PL_FD'].tolist()

LP_Index = v23_df_selected['LP_index'].tolist()

LP_FD = v23_df_selected['LP_FD'].tolist()

LP_Beta = v23_df_selected['LP_beta'].tolist()

PLEC_Index = v23_df_selected['PLEC_Index'].tolist()

PLEC_FD = v23_df_selected['PLEC_FD'] 

PLEC_Index2 = v23_df_selected['PLEC_Index2'] 

PLEC_Expf = v23_df_selected['PLEC_Expf']

spec_type = v23_df_selected['Spec_Type']

name = v23_df_selected['Name']

########################
# create the new xml file comaptible for fermitools 
########################

gll_new_xmlfile = open('./PATH/My Path/gll_pscv26_updated.xml', 'w')
gll_new_xmlfile.write('<source_library title="source library">\n')

for n, sp_t, ra, dec, PEn, PLin, PLfd, lp_in, lp_fd, lp_b, plec_in, plec_prefac, plec_in2, plec_expa in zip(name, spec_type, 
                                                                                                            RA_list, DEC_list, Piv_Energy, 
                                                                                                            PL_Index, PL_FD, LP_Index, 
                                                                                                            LP_FD, LP_Beta, PLEC_Index, 
                                                                                                            PLEC_FD, PLEC_Index2, PLEC_Expf):
  if sp_t==Spec_type_unique[1]:
    gll_new_xmlfile.write('<source name="{0}" type="PointSource">\n'.format(n))
    gll_new_xmlfile.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
    gll_new_xmlfile.write('<spectrum type="LogParabola">\n')
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="norm" scale="1.0" value="{2}"/>\n'.format(max(LP_FD), min(LP_FD), lp_fd))
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="alpha" scale="1.0" value="{2}"/>\n'.format(max(LP_Index), min(LP_Index), lp_in))
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="Eb" scale="1" value="{2}"/>\n'.format(max(Piv_Energy), min(Piv_Energy), PEn))
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="beta" scale="1.0" value="{2}"/>\n'.format(max(LP_Beta), min(LP_Beta), lp_b))
    gll_new_xmlfile.write('</spectrum>\n')
    gll_new_xmlfile.write('<spatialModel type="SkyDirFunction">\n')
    gll_new_xmlfile.write(
        '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
    gll_new_xmlfile.write(
        '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
    gll_new_xmlfile.write('</spatialModel>\n')
    gll_new_xmlfile.write('</source>\n')
  
  elif sp_t==Spec_type_unique[0]:
    gll_new_xmlfile.write('<source name="{0}" type="PointSource">\n'.format(n))
    gll_new_xmlfile.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
    gll_new_xmlfile.write('<spectrum type="PowerLaw">\n')
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="Prefactor" scale="1.0" value="{2}"/>\n'.format(max(PL_FD), min(PL_FD), PLfd))
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="Index" scale="-1.0" value="{2}"/>\n'.format(max(PL_Index), min(PL_Index), PLin))
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="Scale" scale="1.0" value="{2}"/>\n'.format(max(Piv_Energy), min(Piv_Energy), PEn))
    gll_new_xmlfile.write('</spectrum>\n')
    gll_new_xmlfile.write('<spatialModel type="SkyDirFunction">\n')
    gll_new_xmlfile.write(
        '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
    gll_new_xmlfile.write(
        '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
    gll_new_xmlfile.write('</spatialModel>\n')
    gll_new_xmlfile.write('</source>\n')
  else :
    gll_new_xmlfile.write('<source name="{}" type="PointSource">\n'.format(n))
    gll_new_xmlfile.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
    gll_new_xmlfile.write('<spectrum type="PLSuperExpCutoff2">\n')
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="Prefactor" scale="1.0" value="{2}"/>\n'.format(max(PLEC_FD), min(PLEC_FD), plec_prefac))
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="Index1" scale="-1.0" value="{2}"/>\n'.format(max(PLEC_Index), min(PLEC_Index), plec_in))
    gll_new_xmlfile.write(
        '<parameter free="0" max="{0}" min="{1}" name="Scale" scale="1" value="{2}"/>\n'.format(max(Piv_Energy), min(Piv_Energy), PEn))
    gll_new_xmlfile.write(
        '<parameter free="1" max="{0}" min="{1}" name="Expfactor" scale="1.0" value="{2}"/>\n'.format(max(PLEC_Expf), min(PLEC_Expf), plec_expa))
    gll_new_xmlfile.write(
        '<parameter free="0" max="{0}" min="{1}" name="Index2" scale="1.0" value="{2}"/>\n'.format(max(PLEC_Index2), min(PLEC_Index2), plec_in2))
    gll_new_xmlfile.write('</spectrum>\n')
    gll_new_xmlfile.write('<spatialModel type="SkyDirFunction">\n')
    gll_new_xmlfile.write(
        '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
    gll_new_xmlfile.write(
        '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
    gll_new_xmlfile.write('</spatialModel>\n')
    gll_new_xmlfile.write('</source>\n')

gll_new_xmlfile.write('</source_library>')    
gll_new_xmlfile.close()
