'''
for a single mock catalog generation check the 
file 'fermi10yrs4FGLfits.ipynb' under the same folder. 
codes used here are originally from the above notebook and check the notebook for 
detailed analysis 
'''

### imports

import math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import random


from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.io import fits


########

gll_psc_v23 = fits.open('gll_psc_v23.fit')
gll_psc_v23_list1 = gll_psc_v23[1] 

#### prepare to crea the agn dataframe with LogParabola Parameters involved. 

gll_psc_v23_list1_data = gll_psc_v23_list1.data

## prepare the columns for dataframe (LogParabola)

v23LP_Name = gll_psc_v23_list1_data['Source_Name']
v23LP_RA = gll_psc_v23_list1_data['RAJ2000']
v23LP_DEJ = gll_psc_v23_list1_data['DEJ2000']
v23LP_GLON = gll_psc_v23_list1_data['GLON']
v23LP_GLAT = gll_psc_v23_list1_data['GLAT']
v23LP_PEn = gll_psc_v23_list1_data['Pivot_Energy']
v23LP_EnF1000 = gll_psc_v23_list1_data['Energy_Flux100']
v23LP_spectype = gll_psc_v23_list1_data['SpectrumType']
v23LP_FDensity = gll_psc_v23_list1_data['LP_Flux_Density']
v23LP_Index = gll_psc_v23_list1_data['LP_Index']
v23LP_Beta = gll_psc_v23_list1_data['LP_beta']
v23LP_Scurve = gll_psc_v23_list1_data['LP_SigCurv']
v23LP_Variability_Index = gll_psc_v23_list1_data['Variability_Index']
v23LP_Npred = gll_psc_v23_list1_data['Npred']
v23LP_Class1 = gll_psc_v23_list1_data['CLASS1']

v23LP_df = pd.DataFrame(data = v23LP_Name, columns=['N']) ## start of with a single column 

list_of_cols_v23 = [v23LP_GLAT, v23LP_GLON, v23LP_RA, v23LP_DEJ, v23LP_PEn, v23LP_EnF1000, 
                    v23LP_spectype, v23LP_FDensity, v23LP_Index, v23LP_Beta, v23LP_Scurve, 
                    v23LP_Npred, v23LP_Class1]
list_of_cols_strings_v23 = ['GLAT', 'GLON', 'RA', 'DEC', 'Piv_En', 'En_flux_100', 
                            'Spec_Type', 'LP_f_density', 'LP_index', 'LP_beta', 'LP_sig_curv', 'Npred', 'Class1']

for y in range(len(list_of_cols_v23)):
    v23LP_df[list_of_cols_strings_v23[y]] = list_of_cols_v23[y] # complete the creation of dataframe 

### v23LP_df.head(3)     check if necessary 

### look for null values in the dataframe  

# print ('null values in particular col: ', v23LP_df['LP_f_density'].isnull().sum())
# print ('null values in entire dataframe: ', v23LP_df.isnull().values.any())
# print ('total number of null values: ', v23LP_df.isnull().sum().sum())

v23LP_df.dropna(how='any', inplace=True) ## drop the rows with missing values

# create a sub-dataframe from the main where only classified AGNs are present 

# first check the unique elements of column Class1
# Class1_unique = v23LP_df.Class1.unique()
# print ('check unique elements in Class1: ', Class1_unique)

Class1_AGNs = ['fsrq ', 'FSRQ ', 'bll  ', 'BLL  ', 'bcu  ', 'BCU  ', 'rdg  ', 'RDG  ', 'nlsy1', 'NLSY1', 
               'agn  ', 'AGN  ', 'sey  ', 'ssrq '] # beware of capital and small letters 
# based on  these categories, create a dataframe which is unique for only AGNs

v23LP_df_AGNs = v23LP_df[v23LP_df['Class1'].isin(Class1_AGNs)]

LP_df_AGNs_bins = int(np.sqrt(v23LP_df_AGNs.shape[0]))

###################################### 
# Generate the mock catalogue 
######################################

#### First the Log Parabola Parameters 

LP_index_mean = v23LP_df_AGNs['LP_index'].mean()
LP_index_var = v23LP_df_AGNs['LP_index'].var()

LP_beta_mean = v23LP_df_AGNs['LP_beta'].mean()
LP_beta_var = v23LP_df_AGNs['LP_beta'].var()

LP_Fdensity_mean = np.mean(np.log10(v23LP_df_AGNs['LP_f_density'])) # for conversion 10**(LP_Fdensity_mean)
LP_Fdensity_var = np.var(np.log10(v23LP_df_AGNs['LP_f_density']))

LP_PEn_mean = np.mean(v23LP_df_AGNs['Piv_En']) 
LP_PEn_std = np.std(v23LP_df_AGNs['Piv_En'])

from scipy.stats import norm

LP_index_fit_mu, LP_index_fit_std = norm.fit(v23LP_df_AGNs['LP_index'])
LP_beta_fit_mu, LP_beta_fit_std = norm.fit(v23LP_df_AGNs['LP_beta']) 
# definitely LP_beta doesn't follow normal distribution from figure
LP_Fdensity_fit_mu, LP_Fdensity_fit_std = norm.fit(np.log10(v23LP_df_AGNs['LP_f_density']))

LP_df_mockCat_bins = 15000

LPIndex_mock_dist2 = np.random.normal(LP_index_fit_mu, LP_index_fit_std, LP_df_mockCat_bins)
LPBeta_mock_dist2 = np.random.gumbel(LP_beta_mean-0.077, np.sqrt(LP_beta_var)-0.1, size=LP_df_mockCat_bins) # best for beta
LPFDensity_mock_dist1 = 10**(np.random.normal(LP_Fdensity_mean, np.sqrt(LP_Fdensity_var), LP_df_mockCat_bins)) 


from scipy.stats import expon, lognorm

logshape1, logloc1, logscale1 = lognorm.fit(v23LP_df_AGNs['Piv_En'], loc=0)

LPPivEn_mock_dist3 = np.random.lognormal(np.log(logscale1), logshape1, LP_df_mockCat_bins)
LPPivEn_mock_dist3f = LPPivEn_mock_dist3[(LPPivEn_mock_dist3<26000.)] ### select when pivot energy is below 26000 MeV. 
# Comparing with the original distribution.  


###
# AGN Spatial Distribution 
###

sinb = np.random.uniform(-1, 1, LP_df_mockCat_bins) # distribution of sinb from -1, 1
# print ('sinb : ', sinb)
# b will be drawn from sinb distribution 
mock_GLAT = np.rad2deg(np.arcsin(sinb))


gal_long_uniform = np.random.uniform(0.0, 2*np.pi, LP_df_mockCat_bins) # already in radian 
# print ('gal long: ', gal_long)

mock_GLON = np.rad2deg(gal_long_uniform)





E100hist, E100bins, _ = plt.hist(np.log10(v23LP_df_AGNs['En_flux_100']), bins=LP_df_AGNs_bins, 
                                 density=False, alpha=0.8, edgecolor='navy')

plt.clf()
print ('min and max val of lowest bin width: ', E100bins[0], E100bins[-1])
E100hist_list = E100hist.tolist()


#########################################
# Integration for E100 variable 
#########################################

def Integrate(N, a, b, alpha, beta, PivEn, 
              FluxDensity): # number of steps: N, lower and upper limit: a, b 
    value = 0
    value2 = 0
    
    for i in range(1, N+1):
        En1 = a + ( (i-1/2)* ( (b-a)/ N ) )
        x = En1*FluxDensity * ((En1/PivEn)**(- alpha - ( beta *(math.log(En1/PivEn)) ) ))
#         value += LP(En1 , alpha1, beta1, PivEn, FluxDensity)
        value += x
    value2 = ( (b-a)/N ) * value
    return value2


##############################################

num_cats = 10 # num_cats is number of catalog we would like to generate.

for i in range(num_cats):

    #### introduce the poisson noise 

    noise_N = np.random.uniform(0.7, 1.3, LP_df_AGNs_bins)

    #### choose the parameters randomly from the distribution of mock parameters 

    choose_number =  int(len(mock_GLON) * random.choice(np.random.uniform(0.82, 0.92, num_cats)) )

    rand_LPIndex_mock_dist    = random.sample(LPIndex_mock_dist2.tolist(), choose_number)
    rand_LPBeta_mock_dist     = random.sample(LPBeta_mock_dist2.tolist(), choose_number)
    rand_LPFDensity_mock_dist = random.sample(LPFDensity_mock_dist1.tolist(), choose_number)
    rand_LPPivEn_mock_dist    = random.sample(LPPivEn_mock_dist3.tolist(), choose_number)

    rand_GLAT                 = random.sample(mock_GLAT.tolist(), choose_number)
    rand_GLON                 = random.sample(mock_GLON.tolist(), choose_number)



    ###############################~~~~~~~~~~~~~~~~~~~~~~~
    # define the luminosity calculation function 
    ###############################~~~~~~~~~~~~~~~~~~~~~~~

    def simple_lum(n, LPalpha, LPbeta, LPFd, LPPEn, LPglat, LPglon):
        result_list = []
    #     comp_list = [0] * len(E100hist_list)
        final_result_list = []
        c_lp_a   = [] # AGN alpha
        c_lp_b   = [] # AGN beta
        c_lp_F   = [] # flux density
        c_lp_PEn = [] # pivot  energy
        c_GLAT   = [] # lat
        c_GLON   = [] # long
        #### loop over the number of sources (selected in random)
        for x in range(choose_number):
            LP_index = LPalpha[x]
            LP_beta  = LPbeta[x]
            LP_FD    = LPFd[x]
            LP_PEn   = LPPEn[x]
            LP_glat  = LPglat[x]
            LP_glon  = LPglon[x]

            
            
            result   = Integrate(n, 100, 100e3, alpha=LP_index, beta=LP_beta, PivEn=LP_PEn, FluxDensity=LP_FD)
            # n is the number of steps in integration, higher ---> better accuracy
            # unit here is MeV cm^-2 s^-1 :
            result   = result * 1.6e-6 
            # unit here is Erg cm^-2 s^-1:
            # lowest bin -12.378695, highest bin -9.002079  
            if result>=4.181239071515e-13 and result <= 9.95405417351e-10: # 10**(lbin), 10**(hbin)
                result_list.append(result)

                c_lp_a.append(LP_index)
                c_lp_b.append(LP_beta)
                c_lp_F.append(LP_FD)
                c_lp_PEn.append(LP_PEn)
                c_GLAT.append(LP_glat)
                c_GLON.append(LP_glon)
                
                mockE100hist, bins = np.histogram(np.log10(result_list), bins=E100bins, density=False)
                mockE100hist_list = mockE100hist.tolist()
    #         for Ebins in E100bins_notlast:
                for x1, x2, x3 in zip(mockE100hist_list, E100hist_list, noise_N): 
                    if (x1*x3) > x2:
                        print ('!!! mock higher than the real !!!', x1, x2)
                        result_list.pop()
    #                     print ('result_list len: ', len(result_list))
                        final_result_list = result_list[:]
                        c_lp_a.pop()
                        c_lp_b.pop()
                        c_lp_F.pop()
                        c_lp_PEn.pop()
                        c_GLAT.pop()
                        c_GLON.pop()
            
                    else:
                        continue
                        print (':) :) what is happening here :) :)')
                    
    #                 result_list.pop()    
        return final_result_list, c_lp_a, c_lp_b, c_lp_F, c_lp_PEn, c_GLAT, c_GLON


    #### calculate the luminosity 

    final_result_list, c_lp_a, c_lp_b, c_lp_F, c_lp_PEn, c_GLAT, c_GLON = simple_lum(1070, rand_LPIndex_mock_dist, rand_LPBeta_mock_dist, 
                                                                                    rand_LPFDensity_mock_dist, rand_LPPivEn_mock_dist, 
                                                                                    rand_GLAT, rand_GLON)

    # print('check the number of selected sources: \n', len(c_lp_a))

    #### collect the max and min value of each parameter 

    #### Store the min max value of different params for the mock catalog

    c_lp_a_max = max(c_lp_a) + 0.5
    c_lp_a_min = min(c_lp_a) - 0.1

    c_lp_b_max = max(c_lp_b) + 0.1
    c_lp_b_min = min(c_lp_b) - 0.1

    c_lp_FD_max = max(c_lp_F)*5.
    c_lp_FD_min = min(c_lp_F)*0.5

    c_lp_PEn_max = max(c_lp_PEn) + 300. 
    c_lp_PEn_min = min(c_lp_PEn) - 100. 


    c_ra_cord_list = []
    c_dec_cord_list = []

    for clo, cla in zip(c_GLON, c_GLAT):
        c_icrs1_c  = SkyCoord(l=clo*u.degree, b=cla*u.degree, frame='galactic')
        c_ra_cord_list.append(c_icrs1_c.fk5.ra.degree)
        c_dec_cord_list.append(c_icrs1_c.fk5.dec.degree)



    ####################################~~~~~~~~~
    # Create the Source Catalog .xml File
    ####################################~~~~~~~~~


    mock_source_num = [i for i in range(len(c_lp_a))]

    # number of sources in original catalogue 3502. we make +- 2% limit on the number of sources    
    print ('number of sources: ', len(c_lp_a))

    if (len(c_lp_a) <= 3572) and (len(c_lp_a) >= 3432): 


        mock_xmlfile = open('check_mock_4FGL_V23_RA_DEC_%d.xml' %(i), 'w')
        mock_xmlfile.write('<source_library title="source library">\n')
        for n, al, be, pivE, ra, dec, num in zip(c_lp_F, c_lp_a, c_lp_b, c_lp_PEn, c_ra_cord_list, 
                                                c_dec_cord_list, mock_source_num):
            
            mock_xmlfile.write('<source name="LogParabola_source{0}" type="PointSource">\n'.format(num))
            mock_xmlfile.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
            mock_xmlfile.write('<spectrum type="LogParabola">\n')
            mock_xmlfile.write(
                '<parameter free="1" max="{0}" min="{1}" name="norm" scale="1.0" value="{2}"/>\n'.format(c_lp_FD_max, c_lp_FD_min, n))
            mock_xmlfile.write(
                '<parameter free="1" max="{0}" min="{1}" name="alpha" scale="1.0" value="{2}"/>\n'.format(c_lp_a_max, c_lp_a_min, al))
            mock_xmlfile.write(
                '<parameter free="1" max="{0}" min="{1}" name="Eb" scale="1" value="{2}"/>\n'.format(c_lp_PEn_max, c_lp_PEn_min, pivE))
            mock_xmlfile.write(
                '<parameter free="1" max="{0}" min="{1}" name="beta" scale="1.0" value="{2}"/>\n'.format(c_lp_b_max, c_lp_b_min, be))
            mock_xmlfile.write('</spectrum>\n')
            mock_xmlfile.write('<spatialModel type="SkyDirFunction">\n')
            mock_xmlfile.write(
                '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
            mock_xmlfile.write(
                '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
            mock_xmlfile.write('</spatialModel>\n')
            mock_xmlfile.write('</source>\n')
            
        mock_xmlfile.write('</source_library>')    
        mock_xmlfile.close() 
        print ('catalog generation finsihed: ', i)
        print ('taking a break for 3 seconds')
        time.sleep(3)   
    else:
        continue
        print ('shouldnt reach here')    