'''
for a single mock catalog generation check the 
file 'fermi10yrs4FGLfits.ipynb' under the same folder. 
codes used here are originally from the above notebook and check the notebook for 
detailed analysis 

similar to 'generate_agn_cats.py' but used for Pulsars with PLEC parametrization. 

'''

# imports

from scipy.optimize import curve_fit
from scipy.stats import cauchy
from scipy.stats import expon, lognorm
import math
import time
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

# prepare to crea the psr dataframe with PLEC Parameters involved.

gll_psc_v23_list1_data = gll_psc_v23_list1.data

v23LP_Name = gll_psc_v23_list1_data['Source_Name']
v23LP_RA = gll_psc_v23_list1_data['RAJ2000']
v23LP_DEJ = gll_psc_v23_list1_data['DEJ2000']
v23LP_GLON = gll_psc_v23_list1_data['GLON']
v23LP_GLAT = gll_psc_v23_list1_data['GLAT']
v23LP_PEn = gll_psc_v23_list1_data['Pivot_Energy']
v23LP_F1000 = gll_psc_v23_list1_data['Flux1000']
v23LP_EnF1000 = gll_psc_v23_list1_data['Energy_Flux100']
v23LP_spectype = gll_psc_v23_list1_data['SpectrumType']
v23PLEC_Index = gll_psc_v23_list1_data['PLEC_Index']
v23PLEC_ExpIndex = gll_psc_v23_list1_data['PLEC_Exp_Index']
v23PLEC_FDensity = gll_psc_v23_list1_data['PLEC_Flux_Density']
v23PLEC_ExpFactor = gll_psc_v23_list1_data['PLEC_Expfactor']
v23PLEC_SigCurv = gll_psc_v23_list1_data['PLEC_SigCurv']
v23LP_Npred = gll_psc_v23_list1_data['Npred']
v23LP_Class1 = gll_psc_v23_list1_data['CLASS1']


# get started with the dataframe
v23PLEC_df = pd.DataFrame(data=v23LP_Name, columns=['N'])

list_of_cols_v23_plec = [v23LP_GLAT, v23LP_GLON, v23LP_RA, v23LP_DEJ, v23LP_PEn, v23LP_F1000, v23LP_EnF1000,
                         v23LP_spectype, v23PLEC_FDensity, v23PLEC_Index, v23PLEC_ExpIndex,
                         v23PLEC_ExpFactor, v23PLEC_SigCurv, v23LP_Npred, v23LP_Class1]
list_of_cols_strings_v23_plec = ['GLAT', 'GLON', 'RA', 'DEC', 'Piv_En', 'Flux1000', 'En_flux_100',
                                 'Spec_Type', 'PLEC_f_density', 'PLEC_Index', 'PLEC_ExpIndex',
                                 'PLEC_ExpFac', 'PLEC_Sigma', 'Npred', 'Class1']

for pl in range(len(list_of_cols_v23_plec)):
    # finish creating the complete data-frame
    v23PLEC_df[list_of_cols_strings_v23_plec[pl]] = list_of_cols_v23_plec[pl]


# create a sub-dataframe from the main where only classified PSRs are present

Class1_PSRs = ['PSR  ', 'psr  ']

v23PLEC_df_PSR = v23PLEC_df[v23PLEC_df['Class1'].isin(Class1_PSRs)]

PLEC_df_bins = int(np.sqrt(len(v23PLEC_df_PSR)))

######################################
# Generate the mock catalogue
######################################

# First the PLEC Parameters

PLEC_a_mean = v23PLEC_df_PSR['PLEC_ExpFac'].mean()
PLEC_a_std = v23PLEC_df_PSR['PLEC_ExpFac'].std()

PLEC_b_mean = v23PLEC_df_PSR['PLEC_ExpIndex'].mean()
PLEC_b_std = v23PLEC_df_PSR['PLEC_ExpIndex'].std()

PLEC_PEn_mean = v23PLEC_df_PSR['Piv_En'].mean()
PLEC_PEn_std = v23PLEC_df_PSR['Piv_En'].std()

PLEC_index_mean = v23PLEC_df_PSR['PLEC_Index'].mean()
PLEC_index_std = v23PLEC_df_PSR['PLEC_Index'].std()

PLEC_FDensity_mean = np.mean(np.log10(v23PLEC_df_PSR['PLEC_f_density']))
PLEC_FDensity_std = np.std(np.log10(v23PLEC_df_PSR['PLEC_f_density']))

PLEC_df_mock_cat_bins = 800


PLECIndex_mock_dist1 = np.random.normal(
    PLEC_index_mean, PLEC_index_std, PLEC_df_mock_cat_bins)
# index minimum value is 0, so we select array elements based on condition

PLECIndex_mock_distf = PLECIndex_mock_dist1[np.where(
    (PLECIndex_mock_dist1 > 0.) & (PLECIndex_mock_dist1 < 2.9))]


PLECFden_mock_dist1 = 10**(np.random.normal(PLEC_FDensity_mean,
                                            PLEC_FDensity_std, PLEC_df_mock_cat_bins))

PLECb_mock_dist1 = np.random.normal(
    PLEC_b_mean, PLEC_b_std, PLEC_df_mock_cat_bins)


# try as before lognorm distribution

logshapePL_PEn, loglocPL_PEn, logscalePL_PEn = lognorm.fit(
    v23PLEC_df_PSR['Piv_En'], loc=0)
# print('check fit shape, loc and scale, log(scale) Pivot E: ', logshapePL_PEn, loglocPL_PEn,
#       logscalePL_PEn, np.log(logscalePL_PEn))

PLECPivEn_mock_dist3 = np.random.lognormal(
    np.log(logscalePL_PEn), logshapePL_PEn, PLEC_df_mock_cat_bins)


# try again lognorm for parameter a

logshapePL_Exp_a, loglocPL_Exp_a, logscalePL_Exp_a = lognorm.fit(
    v23PLEC_df_PSR['PLEC_ExpFac'], loc=0.02)

PLECExp_a_mock_dist3 = np.random.lognormal(
    np.log(logscalePL_Exp_a), logshapePL_Exp_a, PLEC_df_mock_cat_bins)


cauchy_PLEC_b_scale1 = 0.006  # this value is result of trial and error
cauchy_PLECb_loc = 0.666

PLECExp_b_mock_dist3 = cauchy.rvs(
    cauchy_PLECb_loc, cauchy_PLEC_b_scale1, size=PLEC_df_mock_cat_bins)
PLECExp_b_mock_dist3f = PLECExp_b_mock_dist3[(
    PLECExp_b_mock_dist3 > 0.4) & (PLECExp_b_mock_dist3 < 0.95)]


######################################
# PSR Spatial Distribution (Complicated)
######################################

# from uniform but sorted (alternative np.linspace)
pleclatcoord = sorted(np.random.uniform(-72, 72, 300))


def gauss2mix(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))


nLAT, binsLAT, _ = plt.hist(v23PLEC_df_PSR['GLAT'], bins=PLEC_df_bins, label='histogram', alpha=0.7,
                            density=True)

plt.clf()

bin_centerPLECLAT = (binsLAT[:-1] + binsLAT[1:])/2

# print('bin counts: ', nLAT)
# print('bins: ', binsLAT)

p0lat = [5e-2, 0, 2, 0.01, 0, 17]  # starting values of fit parameters

params, params_cov = curve_fit(gauss2mix, bin_centerPLECLAT, nLAT, p0=p0lat)

# print ('check fitted params 1st and 2nd gaussian : ', params, params[0:3], params[3:])


def firstgauss(x, p1):
    a1, mu1, sigma1 = p1
    return a1*np.exp(-(x-mu1)**2/(2.*sigma1**2))


def secondgauss(x, p2):
    a2, mu2, sigma2 = p2
    return a2*np.exp(-(x-mu2)**2/(2.*sigma2**2))


# ### get the fitted curve
his_fitPLECLAT = gauss2mix(pleclatcoord, *params)
first_hist = firstgauss(pleclatcoord, params[0:3])
second_hist = secondgauss(pleclatcoord, params[3:])


plecloncoord = sorted(np.random.uniform(0, 360, 300))
nLON, binsLON, _ = plt.hist(v23PLEC_df_PSR['GLON'], bins=PLEC_df_bins, label='histogram', alpha=0.7,
                            density=True)

plt.clf()
bin_centerPLECLON = (binsLON[:-1] + binsLON[1:])/2

# print('bin counts LON: ', nLON)
# print('bins LON: ', binsLON)


def gaussoffset2mix(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    g1 = A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))
    g2 = A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
    return g1 + g2


p0lon = [2e-2, -90, 100, 8e-3, 330, 70]  # starting values of fit parameters

paramsLON, params_covLON = curve_fit(
    gaussoffset2mix, bin_centerPLECLON, nLON, p0=p0lon)


def firstgaussoffset(x, p1):
    a1, mu1, sigma1 = p1
    return a1*np.exp(-(x-mu1)**2/(2.*sigma1**2))


def secondgaussoffset(x, p2):
    a2, mu2, sigma2 = p2
    return a2*np.exp(-(x-mu2)**2/(2.*sigma2**2))


# get the fitted curve
his_fitPLECLON = gaussoffset2mix(plecloncoord, *paramsLON)
first_histLON = firstgaussoffset(plecloncoord, paramsLON[0:3])
second_histLON = secondgaussoffset(plecloncoord, paramsLON[3:])


# # mock_LON_psr = random.sample()


possible_PSR_lat1 = np.random.normal(-0.5,  3.56443164, 550)

possible_PSR_lat1_selected = np.random.choice(possible_PSR_lat1, 430)
possible_PSR_lat2 = np.random.normal(2.16, 2.68265452e+01, 550)
# possible_PSR_lat2_selected = random.sample(possible_PSR_lat2.tolist(), 200)
possible_PSR_lat2_selected = np.random.choice(possible_PSR_lat2, 430)
# plt.hist(possible_PSR_lat2, bins=binsLAT, density=True, alpha=0.6)

total_hist = (possible_PSR_lat1) + (possible_PSR_lat2)
total_histLAT1 = np.concatenate(
    (possible_PSR_lat1_selected, possible_PSR_lat2_selected))
total_histLAT1f = total_histLAT1[(
    total_histLAT1 > -80.) & (total_histLAT1 < 80.)]

num_select = 5680
seed1 = 16
np.random.seed(seed1)


def psrLON1(num_select):
    possible_PSR_LON1 = np.random.normal(-2.19250209e+02,
                                         1.44911680e+02, num_select)
    possible_PSR_LON1_selected = np.random.choice(
        possible_PSR_LON1, int(num_select*0.93))
    return possible_PSR_LON1_selected


def psrLON2(num_select):
    possible_PSR_LON2 = np.random.normal(
        3.21788428e+02, 4.96174001e+01, num_select)
    possible_PSR_LON2_selected = np.random.choice(
        possible_PSR_LON2, int(num_select*0.07))
    return possible_PSR_LON2_selected


total_hist_LON = np.concatenate((psrLON1(num_select), psrLON2(num_select)))
total_histLONf = total_hist_LON[(
    total_hist_LON > -0.) & (total_hist_LON < 360.)]


E100histPLEC, E100binsPLEC, _ = plt.hist(np.log10(v23PLEC_df_PSR['En_flux_100']), bins=PLEC_df_bins,
                                         density=False, alpha=0.5, edgecolor='navy')

plt.clf()


E100histPLEC_list = E100histPLEC.tolist()

#########################################
# Integration for E100 variable
#########################################


def IntegratePLEC(N, E1, E2, PivEn, fD, gamma, a, b):
    value = 0.
    value1 = 0.
    for i in range(1, N+1):
        En1 = E1 + ((i-1/2) * ((E2-E1) / N))
        x = En1 * fD * ((En1/PivEn)**(-gamma)) * \
            math.exp(a * (PivEn**b - En1**b))
        value += x
    value1 = ((E2-E1)/N) * value
    return value1


num_cats = 10

for i in range(num_cats):

    # np.random.seed(30) # change seed for different run
    noise_N_PLEC = np.random.uniform(0.7, 1.3, PLEC_df_bins)  # introduce the poisson noise

    choose_number = int(len(total_histLONf) * random.choice(np.random.uniform(0.82, 0.92, num_cats)))

    rand_PLEC_f_density = random.sample(PLECFden_mock_dist1.tolist(), choose_number)
    rand_PLEC_index = random.sample(PLECIndex_mock_distf.tolist(), choose_number)
    rand_PLEC_Exp_b = random.sample(PLECExp_b_mock_dist3f.tolist(), choose_number)
    rand_PLEC_Exp_a = random.sample(PLECExp_a_mock_dist3.tolist(), choose_number)
    rand_PLEC_PivEn = random.sample(PLECPivEn_mock_dist3.tolist(), choose_number)

    rand_GLAT_PLEC = random.sample(total_histLAT1f.tolist(), choose_number)
    rand_GLON_PLEC = random.sample(total_histLONf.tolist(), choose_number)


    def simple_lum_plec(n, PLEC_Gamma, PLEC_Exp_a, PLEC_Exp_b, PLEC_f_density, PLEC_PivEn, GLAT_PLEC, GLON_PLEC):
        result_list = []
    #     comp_list = [0] * len(E100hist_list)
        f_result_list = []
        c_plec_i = []  # PLEC index
        c_plec_Expa = []  # PLEC_Exp_a
        c_plec_Expb = []  # PLEC_Exp_b
        c_plec_FD = []  # plec fd
        c_plec_PEn = []  # pivot  energy
        c_GLAT_plec = []  # lat
        c_GLON_plec = []  # long

        for x in range(choose_number):
            PLEC_index = PLEC_Gamma[x]
            PLEC_Ea = PLEC_Exp_a[x]
            PLEC_Eb = PLEC_Exp_b[x]
            PLEC_FD = PLEC_f_density[x]
            PLEC_PEn = PLEC_PivEn[x]
            PLEC_glat = GLAT_PLEC[x]
            PLEC_glon = GLON_PLEC[x]

            result = IntegratePLEC(n, 100, 100e3, PLEC_PEn,
                                PLEC_FD, PLEC_index, PLEC_Ea, PLEC_Eb)
            # unit here is MeV cm^-2 s^-1 :
            result = result * 1.6e-6
            # unit here is Erg cm^-2 s^-1:
            if result >= 6.67298272e-13 and result <= 9.3779735e-09:
                result_list.append(result)

                c_plec_i.append(PLEC_index)
                c_plec_Expa.append(PLEC_Ea)
                c_plec_FD.append(PLEC_FD)
                c_plec_PEn.append(PLEC_PEn)
                c_plec_Expb.append(PLEC_Eb)
                c_GLAT_plec.append(PLEC_glat)
                c_GLON_plec.append(PLEC_glon)

                mockE100hist_plec, bins_plec = np.histogram(np.log10(result_list), bins=E100binsPLEC,
                                                            density=False)
                mockE100hist_plec_list = mockE100hist_plec.tolist()
    #         for Ebins in E100bins_notlast:
                for x1, x2, x3 in zip(mockE100hist_plec_list, E100histPLEC_list, noise_N_PLEC):
                    if (x1*x3) > x2:
                        print('!!! mock higher than the real !!!', x1, x2)
                        result_list.pop()
    #                     print ('result_list len: ', len(result_list))
                        f_result_list = result_list[:]
                        c_plec_i.pop()
                        c_plec_Expa.pop()
                        c_plec_Expb.pop()
                        c_plec_FD.pop()
                        c_plec_PEn.pop()
                        c_GLAT_plec.pop()
                        c_GLON_plec.pop()

                    else:
                        continue
                        print(':) :) what is happening here :) :)')

    #                 result_list.pop()
        return f_result_list, c_plec_i, c_plec_Expa, c_plec_Expb, c_plec_FD, c_plec_PEn, c_GLAT_plec, c_GLON_plec


    f_result_list_pl, c_pl_i, c_pl_Expa, c_pl_Expb, c_pl_FD, c_pl_PEn, c_GLAT_pl, c_GLON_pl = simple_lum_plec(1000, rand_PLEC_index, rand_PLEC_Exp_a,
                                                                                                            rand_PLEC_Exp_b, rand_PLEC_f_density, 
                                                                                                            rand_PLEC_PivEn, rand_GLAT_PLEC, 
                                                                                                            rand_GLON_PLEC)



    plec_ra_cord_list = []
    plec_dec_cord_list = []

    mock_source_num_plec = [i for i in range(len(c_pl_i))]



    for plo, pla in zip(c_GLON_pl, c_GLAT_pl):
        c_icrs1_pl  = SkyCoord(l=plo*u.degree, b=pla*u.degree, frame='galactic')
        plec_ra_cord_list.append(c_icrs1_pl.fk5.ra.degree)
        plec_dec_cord_list.append(c_icrs1_pl.fk5.dec.degree)


    #### mock cat  for pulsar 
    ##### first check max and min values of each params; 






    c_pl_i_max, c_pl_i_min =  max(c_pl_i)+0.2, min(c_pl_i)-0.2
    c_pl_Expa_max, c_pl_Expa_min = max(c_pl_Expa)+0.1, min(c_pl_Expa)-0.1
    c_pl_Expb_max, c_pl_Expb_min = max(c_pl_Expb)+0.15, min(c_pl_Expb)-0.15
    c_pl_PEn_max, c_pl_PEn_min = max(c_pl_PEn)+300, min(c_pl_PEn)-100
    c_pl_FD_max, c_pl_FD_min = max(c_pl_FD)*3., min(c_pl_FD)*0.5

    # number of sources in original catalogue 254. we make +- 5% limit on the number of sources 
    print ('number of sources: ', len(c_pl_i))

    if (len(c_pl_i) <= 267) and (len(c_pl_i) >= 242): 


        mock_xmlfile_psr = open('check_mock_4FGLV23_psr%d.xml'%(i), 'w')
        mock_xmlfile_psr.write('<source_library title="source library">\n')
        for fac, ga, Ea, Eb, PEn, ra, dec, num in zip(c_pl_FD, c_pl_i, c_pl_Expa, c_pl_Expb, c_pl_PEn,
                                                plec_ra_cord_list, plec_dec_cord_list, mock_source_num_plec):
            
            mock_xmlfile_psr.write('<source name="PLSuperExpCutoff2_source{0}" type="PointSource">\n'.format(num))
            mock_xmlfile_psr.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
            mock_xmlfile_psr.write('<spectrum type="PLSuperExpCutoff2">\n')
            mock_xmlfile_psr.write(
                '<parameter free="1" max="{0}" min="{1}" name="Prefactor" scale="1.0" value="{2}"/>\n'.format(c_pl_FD_max, c_pl_FD_min, fac))
            mock_xmlfile_psr.write(
                '<parameter free="1" max="{0}" min="{1}" name="Index1" scale="1.0" value="{2}"/>\n'.format(c_pl_i_max, c_pl_i_min, ga))
            mock_xmlfile_psr.write(
                '<parameter free="0" max="{0}" min="{1}" name="Scale" scale="1" value="{2}"/>\n'.format(c_pl_PEn_max, c_pl_PEn_min, PEn))
            mock_xmlfile_psr.write(
                '<parameter free="1" max="{0}" min="{1}" name="Expfactor" scale="1.0" value="{2}"/>\n'.format(c_pl_Expa_max, c_pl_Expa_min, Ea))
            mock_xmlfile_psr.write(
                '<parameter free="0" max="{0}" min="{1}" name="Index2" scale="1.0" value="{2}"/>\n'.format(c_pl_Expb_max, c_pl_Expb_min, Eb))
            mock_xmlfile_psr.write('</spectrum>\n')
            mock_xmlfile_psr.write('<spatialModel type="SkyDirFunction">\n')
            mock_xmlfile_psr.write(
                '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
            mock_xmlfile_psr.write(
                '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
            mock_xmlfile_psr.write('</spatialModel>\n')
            mock_xmlfile_psr.write('</source>\n')
            
        mock_xmlfile_psr.write('</source_library>')    
        mock_xmlfile_psr.close()    
        print ('catalog generation finsihed: ', i)
        print ('taking a break for 3 seconds')
        time.sleep(3)
    else:
        continue
        print ('shouldnt reach here')    