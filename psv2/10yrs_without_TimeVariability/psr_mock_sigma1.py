import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import expon, lognorm
from scipy.stats import cauchy
from scipy.optimize import curve_fit

 
from sklearn.linear_model import LinearRegression

from astropy.io import fits


from astropy import units as u
from astropy.coordinates import SkyCoord


from astropy import units as u
from astropy.coordinates import SkyCoord

gll_psc_v22 = fits.open('gll_psc_v27.fit')

gll_psc_v22_list1 = gll_psc_v22[1]

gll_psc_v22_list1_data = gll_psc_v22_list1.data

# check the column names before (First consider only the Log Parabola parameters)
v22LP_Name = gll_psc_v22_list1_data['Source_Name']
v22LP_RA = gll_psc_v22_list1_data['RAJ2000']
v22LP_DEJ = gll_psc_v22_list1_data['DEJ2000']
v22LP_GLON = gll_psc_v22_list1_data['GLON']
v22LP_GLAT = gll_psc_v22_list1_data['GLAT']
v22LP_PEn = gll_psc_v22_list1_data['Pivot_Energy']
v22LP_F1000 = gll_psc_v22_list1_data['Flux1000']
v22LP_EnF1000 = gll_psc_v22_list1_data['Energy_Flux100']
v22LP_spectype = gll_psc_v22_list1_data['SpectrumType']
v22LP_PLIndex = gll_psc_v22_list1_data['PL_Index']
v22LP_FDensity = gll_psc_v22_list1_data['LP_Flux_Density']
v22LP_Index = gll_psc_v22_list1_data['LP_Index']
v22LP_Beta = gll_psc_v22_list1_data['LP_beta']
v22LP_Scurve = gll_psc_v22_list1_data['LP_SigCurv']
v22PLEC_Index = gll_psc_v22_list1_data['PLEC_Index']
v22PLEC_ExpIndex = gll_psc_v22_list1_data['PLEC_Exp_Index']
v22PLEC_FDensity = gll_psc_v22_list1_data['PLEC_Flux_Density']
v22PLEC_ExpFactor = gll_psc_v22_list1_data['PLEC_Expfactor']
v22PLEC_SigCurv = gll_psc_v22_list1_data['PLEC_SigCurv']
v22LP_Signif_Avg = gll_psc_v22_list1_data['Signif_Avg']
v22LP_Npred = gll_psc_v22_list1_data['Npred']
v22LP_Class1 = gll_psc_v22_list1_data['CLASS1']
v22LP_VarIndex = gll_psc_v22_list1_data['Variability_Index']
v22LP_FracVar = gll_psc_v22_list1_data['Frac_Variability']
v22LP_FluxHist = gll_psc_v22_list1_data['Flux_History']


v22LP_Fl_hist_df = pd.DataFrame(v22LP_FluxHist,
                                columns=['08-09', '09-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16', '16-17', '17-18'])

v22LP_FluxHist_av = np.mean(v22LP_FluxHist, axis=1)
v22LP_FluxHist_av_frac = v22LP_FluxHist/v22LP_FluxHist_av[:, None]

v22LP_Fl_hist_frac_df = pd.DataFrame(v22LP_FluxHist_av_frac,
                                     columns=['08-09', '09-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16', '16-17', '17-18'])

frac_flux_history = v22LP_FluxHist.transpose(
)/v22LP_FluxHist.mean(axis=1).transpose()

v22LP_df = pd.DataFrame(data=v22LP_Name, columns=['N'])

list_of_cols_v22 = [v22LP_GLAT, v22LP_GLON, v22LP_RA, v22LP_DEJ, v22LP_PEn, v22LP_F1000, v22LP_EnF1000,
                    v22LP_spectype, v22LP_PLIndex, v22LP_FDensity, v22LP_Index, v22LP_Beta, v22PLEC_Index,
                    v22PLEC_ExpFactor, v22PLEC_ExpIndex, v22PLEC_FDensity, v22LP_Scurve,
                    v22LP_Signif_Avg, v22LP_Npred, v22LP_Class1, v22LP_VarIndex, v22LP_FracVar]
list_of_cols_strings_v22 = ['GLAT', 'GLON', 'RA', 'DEC', 'Piv_En', 'Flux1000', 'En_flux_100',
                            'Spec_Type', 'PL_Index', 'LP_f_density', 'LP_index', 'LP_beta', 'PLEC_Index',
                            'PLEC_ExpFactor', 'PLEC_ExpIndex', 'PLEC_FDensity', 'LP_Sigma',
                            'Signif_Avg', 'Npred', 'Class1', 'VarIndex', 'Frac_Var']

for y in range(len(list_of_cols_v22)):
    v22LP_df[list_of_cols_strings_v22[y]] = list_of_cols_v22[y]

v22LP_df_flux_history = pd.concat([v22LP_df, v22LP_Fl_hist_frac_df], axis=1,)

# drop the NaN in Var Index
v22LP_df_flux_history.dropna(inplace=True)

v22LP_df_psrs = v22LP_df_flux_history[(v22LP_df_flux_history['Class1']=='psr  ') | (v22LP_df_flux_history.Class1=='PSR  ')]
# print (v22LP_df_psrs.shape) 

PLEC_df_bins = int(np.sqrt(len(v22LP_df_psrs)))

PLEC_a_mean = v22LP_df_psrs['PLEC_ExpFactor'].mean()
PLEC_a_std  = v22LP_df_psrs['PLEC_ExpFactor'].std()

PLEC_b_mean = v22LP_df_psrs['PLEC_ExpIndex'].mean()
PLEC_b_std  = v22LP_df_psrs['PLEC_ExpIndex'].std()

PLEC_PEn_mean = v22LP_df_psrs['Piv_En'].mean()
PLEC_PEn_std  = v22LP_df_psrs['Piv_En'].std()

PLEC_index_mean = v22LP_df_psrs['PLEC_Index'].mean()
PLEC_index_std  = v22LP_df_psrs['PLEC_Index'].std()

PLEC_FDensity_mean = np.mean(np.log10(v22LP_df_psrs['PLEC_FDensity']))
PLEC_FDensity_std = np.std(np.log10(v22LP_df_psrs['PLEC_FDensity']))

## psr mock data 

PLEC_df_mock_cat_bins = 1200

# seed1=33
# np.random.seed(seed1)

PLECIndex_mock_dist1 = np.random.normal(PLEC_index_mean, PLEC_index_std, PLEC_df_mock_cat_bins)
### index minimum value is 0, so we select array elements based on condition 
# print (PLECIndex_mock_dist1.shape)
PLECIndex_mock_distf = PLECIndex_mock_dist1[np.where((PLECIndex_mock_dist1>0.) & (PLECIndex_mock_dist1<2.7))]
print (PLECIndex_mock_distf.shape)

PLECFden_mock_dist1  = 10**(np.random.normal(PLEC_FDensity_mean, PLEC_FDensity_std, PLEC_df_mock_cat_bins))

PLECb_mock_dist1     = np.random.normal(PLEC_b_mean, PLEC_b_std, PLEC_df_mock_cat_bins)


#### try as before lognorm distribution 

logshapePL_PEn, loglocPL_PEn, logscalePL_PEn = lognorm.fit(v22LP_df_psrs['Piv_En'], loc=0)
# print ('check fit shape, loc and scale, log(scale) Pivot E: ', logshapePL_PEn, loglocPL_PEn, 
#        logscalePL_PEn, np.log(logscalePL_PEn))

PLECPivEn_mock_dist3 = np.random.lognormal(np.log(logscalePL_PEn), logshapePL_PEn, PLEC_df_mock_cat_bins)
# plt.hist(LPPivEn_mock_dist3, density=True, bins=56, color='navy')

# try again lognorm for parameter a

logshapePL_Exp_a, loglocPL_Exp_a, logscalePL_Exp_a = lognorm.fit(v22LP_df_psrs['PLEC_ExpFactor'], loc=0.02)
# print ('check fit shape, loc and scale, log(scale) Exp a:  ', logshapePL_Exp_a, loglocPL_Exp_a, 
#        logscalePL_Exp_a, np.log(logscalePL_Exp_a))

PLECExp_a_mock_dist3 = np.random.lognormal(np.log(logscalePL_Exp_a), logshapePL_Exp_a, PLEC_df_mock_cat_bins)


cauchy_PLEC_b_scale1 = 0.006 # this value is result of trial and error 
cauchy_PLECb_loc = 0.666

PLECExp_b_mock_dist3 = cauchy.rvs(cauchy_PLECb_loc, cauchy_PLEC_b_scale1, size=PLEC_df_mock_cat_bins)
PLECExp_b_mock_dist3f = PLECExp_b_mock_dist3[(PLECExp_b_mock_dist3>0.4) & (PLECExp_b_mock_dist3<0.95)]
# ## check the cut limits 
# print (len(PLECExp_b_mock_dist3f))

# GLAT and GLON

psr_GLON_arr = v22LP_df_psrs[['GLON']].to_numpy()
psr_GLAT_arr = v22LP_df_psrs[['GLAT']].to_numpy()
# print (psr_GLON_arr.shape)
psr_GLON_arr_where = np.where(psr_GLON_arr <= 180., psr_GLON_arr, psr_GLON_arr-360.)


pleclatcoord = sorted(np.random.uniform(-72, 72, 300)) # from uniform but sorted (alternative np.linspace)


def gauss2mix(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))


nLAT, binsLAT, _ = plt.hist(v22LP_df_psrs['GLAT'], bins=PLEC_df_bins, label='histogram', alpha=0.7, 
                            density=True)

plt.clf()

bin_centerPLECLAT = (binsLAT[:-1] + binsLAT[1:])/2

# print ('bin counts: ', nLAT)
# print ('bins: ', binsLAT)

p0lat = [5e-2, 0, 2, 0.01, 0, 17]
params, params_cov = curve_fit(gauss2mix, bin_centerPLECLAT, nLAT, p0=p0lat)


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


plecloncoord = sorted(np.random.uniform(-180, 180, 300))

nLON, binsLON, _ = plt.hist(psr_GLON_arr_where, bins=PLEC_df_bins, label='histogram', alpha=0.7, 
                            density=True)

plt.clf()


bin_centerPLECLON = (binsLON[:-1] + binsLON[1:])/2


def gaussoffsetLON(x, *p):
    A1, mu1, sigma1 = p
    g1 = A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) 
    return g1 

p0lon = [2e-2, 0, 70]

paramsLON, params_covLON = curve_fit(gaussoffsetLON, bin_centerPLECLON, nLON, p0=p0lon)


def firstgaussoffset(x, p1):
    a1, mu1, sigma1 = p1
    return a1*np.exp(-(x-mu1)**2/(2.*sigma1**2))


his_fitPLECLON = gaussoffsetLON(plecloncoord, *paramsLON)

possible_PSR_lat1 = np.random.normal(-0.5,  3.56443164, 800)
possible_PSR_lat1_selected = np.random.choice(possible_PSR_lat1, 750)

possible_PSR_lat2 = np.random.normal(2.16, 2.68265452e+01, 800)
possible_PSR_lat2_selected = np.random.choice(possible_PSR_lat2, 750)

total_histLAT1 = np.concatenate((possible_PSR_lat1_selected, possible_PSR_lat2_selected))
total_histLAT1f = total_histLAT1[(total_histLAT1>-80.) & (total_histLAT1<80.)]

print ('total_histLAT1f: ', len(total_histLAT1f))

mock_num_lon = 990

def pleclon(locL, scaleL, num):
  possible_PSR_lon1 = np.random.normal(locL, scaleL, size=mock_num_lon)
  possible_psr_GLON_where = np.where(possible_PSR_lon1 > 0., possible_PSR_lon1, possible_PSR_lon1+360.)
  possible_selected_GLON = possible_psr_GLON_where[(possible_psr_GLON_where>0.) & (possible_psr_GLON_where<360.)]
  return possible_selected_GLON

possible_final_mock_PLEC_LON = pleclon(paramsLON[1], paramsLON[2], mock_num_lon)

print ('selected LON: ', len(possible_final_mock_PLEC_LON))

### Integration for PLEC parametrization  

def IntegratePLEC(N, E1, E2, PivEn, fD, 
                 gamma, a, b):
    value = 0.
    value1 = 0. 
    for i in range(1, N+1):
        En1 = E1 + ( (i-1/2)* ( (E2-E1)/ N ) )
        x = En1 * fD * ((En1/PivEn)**(-gamma)) * math.exp(a * (PivEn**b - En1**b))
        value +=x
    value1 = ( (E2-E1)/N ) * value
    return value1

# E100 for psrs

E100hist_psr, E100bins_psr,_ = plt.hist(np.log10(v22LP_df_psrs['En_flux_100']), bins=PLEC_df_bins, 
                                density=False, alpha=0.8, edgecolor='navy', color='chocolate')

plt.clf()

E100hist_list_psr = E100hist_psr.tolist()


selected_hist1_psr_fit = E100hist_psr[2:6]
selected_hist1_psr = E100hist_psr[0:6]
selected_hist1_list_psr_fit = [np.log10(k) for k in selected_hist1_psr_fit]


E100bins_psr_list = E100bins_psr.tolist()
selected_bins1_psr_fit = E100bins_psr[2:6] 
selected_bins1_psr = E100bins_psr[0:6]

selected_bins1_list_psr_fit = [i for i in selected_bins1_psr_fit]
selected_bins1_list_psr = [i for i in selected_bins1_psr]
selected_bins1_list_psr_extended = [selected_bins1_list_psr[0]-selected_bins1_list_psr[1] + selected_bins1_list_psr[0]] + selected_bins1_list_psr

reg_psr = LinearRegression(fit_intercept=True)
reg_psr.fit(np.reshape(selected_bins1_list_psr_fit, (-1, 1)), selected_hist1_list_psr_fit)


hist_vals_psr_check = [(reg_psr.coef_[0] * i + reg_psr.intercept_) for i in selected_bins1_list_psr_fit]
hist_vals_psr_check_extended = [(reg_psr.coef_[0] * i + reg_psr.intercept_) for i in selected_bins1_list_psr_extended]
selected_bins1_psr_extended_arr = np.array(selected_bins1_list_psr_extended)

y_err1sig_psr = selected_bins1_psr_extended_arr.std() * np.sqrt(1/len(selected_bins1_psr_extended_arr) + (selected_bins1_psr_extended_arr - selected_bins1_psr_extended_arr.mean())**2 / np.sum((selected_bins1_psr_extended_arr - selected_bins1_psr_extended_arr.mean())**2))
y_err2sig_psr = (2*selected_bins1_psr_extended_arr.std()) * np.sqrt(1/len(selected_bins1_psr_extended_arr) + (selected_bins1_psr_extended_arr - selected_bins1_psr_extended_arr.mean())**2 / np.sum((selected_bins1_psr_extended_arr - selected_bins1_psr_extended_arr.mean())**2))


check_area1sigl_psr = hist_vals_psr_check_extended - y_err1sig_psr
check_area1sigh_psr = hist_vals_psr_check_extended + y_err1sig_psr
check_area2sigh_psr = hist_vals_psr_check_extended + y_err2sig_psr



check_area1sigh_psr_list = check_area1sigh_psr.tolist()
check_area1sigh_psr_list_pow = [10**i for i in check_area1sigh_psr_list] 

E100hist_list_psr_high = E100hist_list_psr[6:]
E100bins_psr_list_high = E100bins_psr_list[6:]


E100hist_list_psr_combined = check_area1sigh_psr_list_pow + E100hist_list_psr_high
E100bins_psr_list_combined = selected_bins1_list_psr_extended + E100bins_psr_list_high




#rand_PLEC_f_density = random.sample(PLECFden_mock_dist1.tolist(), 590) 
#rand_PLEC_index = random.sample(PLECIndex_mock_distf.tolist(), 590)
#rand_PLEC_Exp_b = random.sample(PLECExp_b_mock_dist3f.tolist(), 590)
#rand_PLEC_Exp_a = random.sample(PLECExp_a_mock_dist3.tolist(), 590)
#rand_PLEC_PivEn = random.sample(PLECPivEn_mock_dist3.tolist(), 590)

#rand_GLAT_PLEC = random.sample(total_histLAT1f.tolist(), 590)
#rand_GLON_PLEC = random.sample(possible_final_mock_PLEC_LON.tolist(), 590)



#+++++++++++++++++++++++++++++++
#+
#+++++++++++++++++++++++++++++++

num_cats = 5
source_num = 0
print ('reached before for loop num cats')

for i in range(num_cats):

    noise_N_PLEC = np.random.uniform(0.7, 1.3, PLEC_df_bins)
    noise_N_PLEC_combined = np.random.uniform(0.8, 1.2, PLEC_df_bins+1)

    choose_number_psr = int(len(possible_final_mock_PLEC_LON) * random.choice(np.random.uniform(0.85, 0.95, num_cats)))
    print ('choose number', choose_number_psr)	

    #### randomly select parameters from the distribution [PLEC]

    rand_PLEC_f_density = random.sample(PLECFden_mock_dist1.tolist(), choose_number_psr) 
    rand_PLEC_index = random.sample(PLECIndex_mock_distf.tolist(), choose_number_psr)
    rand_PLEC_Exp_b = random.sample(PLECExp_b_mock_dist3f.tolist(), choose_number_psr)
    rand_PLEC_Exp_a = random.sample(PLECExp_a_mock_dist3.tolist(), choose_number_psr)
    rand_PLEC_PivEn = random.sample(PLECPivEn_mock_dist3.tolist(), choose_number_psr)

    rand_GLAT_PLEC = random.sample(total_histLAT1f.tolist(), choose_number_psr)
    rand_GLON_PLEC = random.sample(possible_final_mock_PLEC_LON.tolist(), choose_number_psr)

    def simple_lum_plec(n, PLEC_i, PLEC_a, PLEC_b, PLEC_FluxD, PLEC_PivEn, PLEC_lat, PLEC_lon):
        result_list = []
        f_result_list = []
        c_plec_i   = [] # PLEC index
        c_plec_Expa   = [] # PLEC_Exp_a
        c_plec_Expb   = [] # PLEC_Exp_b
        c_plec_FD   = [] # plec fd
        c_plec_PEn = [] # pivot  energy
        c_GLAT_plec   = [] # lat
        c_GLON_plec   = [] # long

        for x in range(choose_number_psr):
            PLEC_index = PLEC_i[x]
            PLEC_Ea   = PLEC_a[x]
            PLEC_Eb   = PLEC_b[x]
            PLEC_FD   = PLEC_FluxD[x]
            PLEC_PEn  = PLEC_PivEn[x]
            PLEC_glat = PLEC_lat[x]
            PLEC_glon  = PLEC_lon[x]

        
        
            result_psr   = IntegratePLEC(n, 100, 100e3, PivEn = PLEC_PEn, fD = PLEC_FD, gamma = PLEC_index, a = PLEC_Ea, b = PLEC_Eb)
            # unit here is MeV cm^-2 s^-1 :
            result_psr   = result_psr * 1.6e-6 
            # unit here is Erg cm^-2 s^-1:
            if result_psr>=10**E100bins_psr_list_combined[0] and result_psr < 10**(E100bins_psr_list_combined[-1]):
                result_list.append(result_psr)

                c_plec_i.append(PLEC_index)
                c_plec_Expa.append(PLEC_Ea)
                c_plec_FD.append(PLEC_FD)
                c_plec_PEn.append(PLEC_PEn)
                c_plec_Expb.append(PLEC_Eb)
                c_GLAT_plec.append(PLEC_glat)
                c_GLON_plec.append(PLEC_glon)
            
                mockE100hist_plec, bins_plec = np.histogram(np.log10(result_list), bins=E100bins_psr_list_combined, density=False)
                mockE100hist_plec_list = mockE100hist_plec.tolist()
#               for Ebins in E100bins_notlast:
                for x1, x2, x3 in zip(mockE100hist_plec_list, E100hist_list_psr_combined, noise_N_PLEC_combined): 
                    if (x1*x3) > x2:
                        # print ('!!! mock higher than the real !!!', x1, x2)
                        result_list.pop()
#                       print ('result_list len: ', len(result_list))
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
                        print (':) :) what is happening here :) :)')
                
#                 result_list.pop()    
        return f_result_list, c_plec_i, c_plec_Expa, c_plec_Expb, c_plec_FD, c_plec_PEn, c_GLAT_plec, c_GLON_plec

    f_result_list_pl, c_pl_i, c_pl_Expa, c_pl_Expb, c_pl_FD, c_pl_PEn, c_GLAT_pl, c_GLON_pl = simple_lum_plec(1020, 
                                                                                                          rand_PLEC_index, 
                                                                                                          rand_PLEC_Exp_a, rand_PLEC_Exp_b, 
                                                                                                          rand_PLEC_f_density, rand_PLEC_PivEn, 
                                                                                                          rand_GLAT_PLEC, rand_GLON_PLEC)
    # print ('number of psrs: ', len(c_pl_i))

    psr_ra_cord_list = []
    psr_dec_cord_list = []

    mock_source_num_psr = [i for i in range(len(c_pl_i))]
    print ('source_num: ', len(c_pl_i))	



    for plo, pla in zip(c_GLON_pl, c_GLAT_pl):
        c_icrs1_pl  = SkyCoord(l=plo*u.degree, b=pla*u.degree, frame='galactic')
        psr_ra_cord_list.append(c_icrs1_pl.fk5.ra.degree)
        psr_dec_cord_list.append(c_icrs1_pl.fk5.dec.degree)

    if (len(c_pl_i)<=500 ) and (len(c_pl_i)>=300):     

        source_numbers_check = len(c_pl_Expb)
        print ('sources in file: ', source_numbers_check)

        mock_xmlfile_psr = open('./gen_psr/mock_4FGL_V22_psr_file%dS%d.xml' %(source_num, source_numbers_check), 'w')
        mock_xmlfile_psr.write('<source_library title="source library">\n')
        for fac, ga, Ea, Eb, PEn, ra, dec, num in zip(c_pl_FD, c_pl_i, c_pl_Expa, c_pl_Expb, c_pl_PEn,
                                         psr_ra_cord_list, psr_dec_cord_list, mock_source_num_psr):

            mock_xmlfile_psr.write('<source name="PLSuperExpCutoff2_source{0}" type="PointSource">\n'.format(num))
            mock_xmlfile_psr.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
            mock_xmlfile_psr.write('<spectrum type="PLSuperExpCutoff2">\n')
            mock_xmlfile_psr.write(
                '<parameter free="1" max="{0}" min="{1}" name="Prefactor" scale="1.0" value="{2}"/>\n'.format(max(c_pl_FD), min(c_pl_FD), fac))
            mock_xmlfile_psr.write(
                '<parameter free="1" max="{0}" min="{1}" name="Index1" scale="1.0" value="{2}"/>\n'.format(max(c_pl_i), min(c_pl_i), ga))
            mock_xmlfile_psr.write(
                '<parameter free="0" max="{0}" min="{1}" name="Scale" scale="1" value="{2}"/>\n'.format(max(c_pl_PEn), min(c_pl_PEn), PEn))
            mock_xmlfile_psr.write(
                '<parameter free="1" max="{0}" min="{1}" name="Expfactor" scale="1.0" value="{2}"/>\n'.format(max(c_pl_Expa), min(c_pl_Expa), Ea))
            mock_xmlfile_psr.write(
                '<parameter free="0" max="{0}" min="{1}" name="Index2" scale="1.0" value="{2}"/>\n'.format(max(c_pl_Expb), min(c_pl_Expb), Eb))
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

    else:
        continue
        print ('!!! continue but it should not reach here')
    source_num = source_num + 1
