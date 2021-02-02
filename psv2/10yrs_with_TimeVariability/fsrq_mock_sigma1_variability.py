import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import expon, lognorm
from sklearn.linear_model import LinearRegression

from astropy.io import fits


from astropy import units as u
from astropy.coordinates import SkyCoord

FGL_path = '/d4/CAC/sbhattacharyya/Documents/photon/'

# gll_psc_v22 = fits.open(FGL_path + 'gll_psc_v27.fit')
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

Class1_AGNs = ['fsrq ', 'FSRQ ', 'bll  ', 'BLL  ', 'bcu  ', 'BCU  ', 'rdg  ', 'RDG  ', 'nlsy1', 'NLSY1',
               'agn  ', 'AGN  ', 'sey  ', 'ssrq ']  # beware of capital and small letters
# based on  these categories, create a dataframe which is unique for only AGNs

v22LP_df_AGNs_Flux_History = v22LP_df_flux_history[v22LP_df_flux_history['Class1'].isin(
    Class1_AGNs)]


v22LP_df_fsrq = v22LP_df_AGNs_Flux_History[(v22LP_df_AGNs_Flux_History['Class1']=='fsrq ') | (v22LP_df_AGNs_Flux_History.Class1=='FSRQ ')]
print (v22LP_df_fsrq.shape) 


v22LP_df_bcu = v22LP_df_AGNs_Flux_History[(v22LP_df_AGNs_Flux_History['Class1'] == 'bcu  ') | (
    v22LP_df_AGNs_Flux_History.Class1 == 'BCU  ')]


AGNs_fsrq_bins = int(np.sqrt(v22LP_df_fsrq.shape[0]))
AGNs_bcu_bins = int(np.sqrt(v22LP_df_bcu.shape[0]))

##############################
###### Variability 
###############################

# Consider the High TS Sources, Fit LogNormal Distribution for Yearly Flux / Mean Flux
# Assume all sources follow the same dist
### TS < 100, 100<TS<1000, TS>1000

probable_Signif_AvgHigh_all_sources = v22LP_df_flux_history[(v22LP_df_flux_history['Signif_Avg'] >= np.sqrt(1000))]
# print ('sources with high significance average: ', probable_Signif_AvgHigh_all_sources.shape[0])

SignifAvg_High_fsrq = probable_Signif_AvgHigh_all_sources[(probable_Signif_AvgHigh_all_sources['Class1']=='fsrq ') | (probable_Signif_AvgHigh_all_sources.Class1=='FSRQ ')]
# print ('Signif_Avg High fsrq source only: ', SignifAvg_High_fsrq.shape)

SignifAvg_High_fsrq_year_frac_avg_arr = SignifAvg_High_fsrq[['08-09', '09-10', '10-11', '11-12', 
                                                        '12-13', '13-14', '14-15', '15-16', '16-17', '17-18']].to_numpy()

SignifAvg_High_fsrq_year_frac_avg_arr_flat = SignifAvg_High_fsrq_year_frac_avg_arr.flatten()



LP_df_mockCat_bins_fsrq = 5000

logshape1_fsrq_year_mean, logloc1_fsrq_year_mean, logscale1_fsrq_year_mean = lognorm.fit(SignifAvg_High_fsrq_year_frac_avg_arr_flat, 
                                                                                         loc=0)

year_mean_mock_dist3_fsrq = np.random.lognormal(np.log(logscale1_fsrq_year_mean), logshape1_fsrq_year_mean, 
                                                LP_df_mockCat_bins_fsrq)
year_mean_mock_dist3ffsrq = year_mean_mock_dist3_fsrq[(year_mean_mock_dist3_fsrq<10.)]



### mock distribution mean = 1 (Gulli's Suggestion)
hist_fsrq_mock_variability, bins_fsrq_mock_variability, _ = plt.hist(year_mean_mock_dist3ffsrq, 
                                                                  bins=int(np.sqrt(LP_df_mockCat_bins_fsrq)), 
                                                                  density=True, color='red', alpha=0.6, label='FSRQ Mock')

plt.clf()

hist_fsrq_mock_variability_last = len(hist_fsrq_mock_variability) - np.sum(hist_fsrq_mock_variability[:-1])
# print (last_E100hist_fsrq_variability)

year_mean_mock_dist3ffsrq_list = year_mean_mock_dist3ffsrq.tolist()

new_hist_fsrq_mock_variability = hist_fsrq_mock_variability[:-1] + [hist_fsrq_mock_variability_last]
year_mean_mock_dist3ffsrq_list_new = year_mean_mock_dist3ffsrq_list[:-1] + [len(year_mean_mock_dist3ffsrq_list) - sum(year_mean_mock_dist3ffsrq_list[:-1])]

# print ('check the mean of distrib: ', np.mean(year_mean_mock_dist3ffsrq_list_new))
# print ('check length of var mock list: ', len(year_mean_mock_dist3ffsrq_list_new))

# E100 for FSRQs

E100hist_fsrq, E100bins_fsrq,_ = plt.hist(np.log10(v22LP_df_fsrq['En_flux_100']), bins=AGNs_fsrq_bins, 
                                density=False, alpha=0.8, edgecolor='navy', color='purple')

plt.clf()

E100hist_list_fsrq = E100hist_fsrq.tolist()


selected_hist1_fsrq_fit = E100hist_fsrq[2:5]
selected_hist1_fsrq = E100hist_fsrq[0:5]
selected_hist1_list_fsrq_fit = [np.log10(k) for k in selected_hist1_fsrq_fit]


E100bins_fsrq_list = E100bins_fsrq.tolist()
selected_bins1_fsrq_fit = E100bins_fsrq[2:5] 
selected_bins1_fsrq = E100bins_fsrq[0:5]

selected_bins1_list_fsrq_fit = [i for i in selected_bins1_fsrq_fit]
selected_bins1_list_fsrq = [i for i in selected_bins1_fsrq]
selected_bins1_list_fsrq_extended = [selected_bins1_list_fsrq[0]-selected_bins1_list_fsrq[1] + selected_bins1_list_fsrq[0]] + selected_bins1_list_fsrq
# print ('check extended bins fsrq list: ', selected_bins1_list_fsrq_extended)


from sklearn.linear_model import LinearRegression
reg_fsrq = LinearRegression(fit_intercept=True)
reg_fsrq.fit(np.reshape(selected_bins1_list_fsrq_fit, (-1, 1)), selected_hist1_list_fsrq_fit)

hist_vals_fsrq_check = [(reg_fsrq.coef_[0] * i + reg_fsrq.intercept_) for i in selected_bins1_list_fsrq_fit]
hist_vals_fsrq_check_extended = [(reg_fsrq.coef_[0] * i + reg_fsrq.intercept_) for i in selected_bins1_list_fsrq_extended]
selected_bins1_fsrq_extended_arr = np.array(selected_bins1_list_fsrq_extended)

y_err1sig_fsrq = selected_bins1_fsrq_extended_arr.std() * np.sqrt(1/len(selected_bins1_fsrq_extended_arr) + (selected_bins1_fsrq_extended_arr - selected_bins1_fsrq_extended_arr.mean())**2 / np.sum((selected_bins1_fsrq_extended_arr - selected_bins1_fsrq_extended_arr.mean())**2))
y_err2sig_fsrq = (2*selected_bins1_fsrq_extended_arr.std()) * np.sqrt(1/len(selected_bins1_fsrq_extended_arr) + (selected_bins1_fsrq_extended_arr - selected_bins1_fsrq_extended_arr.mean())**2 / np.sum((selected_bins1_fsrq_extended_arr - selected_bins1_fsrq_extended_arr.mean())**2))

# # # hist_vals = [reg.coef_ * i + reg.intercept_ for i in selected_bins1_list]
# # print ('fitted hist vals: ', hist_vals_bll)

# plt.plot(selected_bins1_list_fsrq_fit, np.power(10, hist_vals_fsrq_check), color='red', label='Regression Fit Line')
# # # # plt.plot(selected_bins1_list_bll, np.power(10, hist_vals_bll_check), color='orange', label='Regression Fit Line (c)')

check_area1sigl_fsrq = hist_vals_fsrq_check_extended - y_err1sig_fsrq
check_area1sigh_fsrq = hist_vals_fsrq_check_extended + y_err1sig_fsrq
check_area2sigh_fsrq = hist_vals_fsrq_check_extended + y_err2sig_fsrq

# # # ax1.fill_between(selected_bins1_list_bll, np.power(10, check_area1), np.power(10, check_area2), alpha=0.2)

# plt.plot(selected_bins1_list_fsrq_extended, np.power(10, check_area1sigh_fsrq), color='orange', label=r'Regression Fit Line $+ 1 \sigma$')
# plt.plot(selected_bins1_list_fsrq_extended, np.power(10, check_area2sigh_fsrq), color='salmon', label=r'Regression Fit Line $+ 2 \sigma$')



check_area1sigh_fsrq_list = check_area1sigh_fsrq.tolist()
check_area1sigh_fsrq_list_pow = [10**i for i in check_area1sigh_fsrq_list] 

E100hist_list_fsrq_high = E100hist_list_fsrq[5:]
E100bins_fsrq_list_high = E100bins_fsrq_list[5:]


E100hist_list_fsrq_combined = check_area1sigh_fsrq_list_pow + E100hist_list_fsrq_high
E100bins_fsrq_list_combined = selected_bins1_list_fsrq_extended + E100bins_fsrq_list_high



# E100 for BCUs

E100hist_bcu, E100bins_bcu, _ = plt.hist(np.log10(v22LP_df_bcu['En_flux_100']), bins=AGNs_bcu_bins,
                                         density=False, alpha=0.8, edgecolor='navy', color='magenta')

plt.clf()

E100hist_list_bcu = E100hist_bcu.tolist()


selected_hist1_bcu_fit = E100hist_bcu[6:11]
selected_hist1_bcu = E100hist_bcu[0:11]
selected_hist1_list_bcu_fit = [np.log10(k) for k in selected_hist1_bcu_fit]


E100bins_bcu_list = E100bins_bcu.tolist()
selected_bins1_bcu_fit = E100bins_bcu[6:11]
selected_bins1_bcu = E100bins_bcu[0:11]

# # print ('shape of selected bins: ', selected_bins1_bll_fit.shape)
# # print ('selected bins: ', selected_bins1_bll_fit)
selected_bins1_list_bcu_fit = [i for i in selected_bins1_bcu_fit]
selected_bins1_list_bcu = [i for i in selected_bins1_bcu]
selected_bins1_list_bcu_extended = [
    selected_bins1_list_bcu[0]-selected_bins1_list_bcu[1] + selected_bins1_list_bcu[0]] + selected_bins1_list_bcu


reg_bcu = LinearRegression(fit_intercept=True)
reg_bcu.fit(np.reshape(selected_bins1_list_bcu_fit,
                       (-1, 1)), selected_hist1_list_bcu_fit)
# # # # reg.fit(selected_bins1_list, selected_hist1_list)
# # # print ('check fit values: intercept: ', reg_bll.intercept_)
# print ('check fit values: coeff and intercept: ', reg_bcu.coef_[0], reg_bcu.intercept_)

hist_vals_bcu_check = [(reg_bcu.coef_[0] * i + reg_bcu.intercept_)
                       for i in selected_bins1_list_bcu_fit]
hist_vals_bcu_check_extended = [
    (reg_bcu.coef_[0] * i + reg_bcu.intercept_) for i in selected_bins1_list_bcu_extended]
selected_bins1_bcu_extended_arr = np.array(selected_bins1_list_bcu_extended)

y_err1sig_bcu = selected_bins1_bcu_extended_arr.std() * np.sqrt(1/len(selected_bins1_bcu_extended_arr) + (selected_bins1_bcu_extended_arr -
                                                                                                          selected_bins1_bcu_extended_arr.mean())**2 / np.sum((selected_bins1_bcu_extended_arr - selected_bins1_bcu_extended_arr.mean())**2))
y_err2sig_bcu = (2*selected_bins1_bcu_extended_arr.std()) * np.sqrt(1/len(selected_bins1_bcu_extended_arr) + (selected_bins1_bcu_extended_arr -
                                                                                                              selected_bins1_bcu_extended_arr.mean())**2 / np.sum((selected_bins1_bcu_extended_arr - selected_bins1_bcu_extended_arr.mean())**2))

# # # # hist_vals = [reg.coef_ * i + reg.intercept_ for i in selected_bins1_list]
# # # print ('fitted hist vals: ', hist_vals_bll)


check_area1sigl_bcu = hist_vals_bcu_check_extended - y_err1sig_bcu
check_area1sigh_bcu = hist_vals_bcu_check_extended + y_err1sig_bcu
check_area2sigh_bcu = hist_vals_bcu_check_extended + y_err2sig_bcu


check_area1sigh_bcu_list = check_area1sigh_bcu.tolist()
check_area1sigh_bcu_list_pow = [10**i for i in check_area1sigh_bcu_list]

E100hist_list_bcu_high = E100hist_list_bcu[11:]
E100bins_bcu_list_high = E100bins_bcu_list[11:]


E100hist_list_bcu_combined = check_area1sigh_bcu_list_pow + E100hist_list_bcu_high
E100bins_bcu_list_combined = selected_bins1_list_bcu_extended + E100bins_bcu_list_high


LP_index_mean_fsrq = v22LP_df_fsrq['LP_index'].mean()
LP_index_var_fsrq = v22LP_df_fsrq['LP_index'].var()

LP_beta_mean_fsrq = v22LP_df_fsrq['LP_beta'].mean()
LP_beta_var_fsrq = v22LP_df_fsrq['LP_beta'].var()

LP_Fdensity_mean_fsrq = np.mean(np.log10(v22LP_df_fsrq['LP_f_density'])) # for conversion 10**(LP_Fdensity_mean)
LP_Fdensity_var_fsrq = np.var(np.log10(v22LP_df_fsrq['LP_f_density']))

LP_PEn_mean_fsrq = np.mean(v22LP_df_fsrq['Piv_En']) 
LP_PEn_std_fsrq = np.std(v22LP_df_fsrq['Piv_En'])


LP_index_mean_bcu = v22LP_df_bcu['LP_index'].mean()
LP_index_var_bcu = v22LP_df_bcu['LP_index'].var()

LP_beta_mean_bcu = v22LP_df_bcu['LP_beta'].mean()
LP_beta_var_bcu = v22LP_df_bcu['LP_beta'].var()

# for conversion 10**(LP_Fdensity_mean)
LP_Fdensity_mean_bcu = np.mean(np.log10(v22LP_df_bcu['LP_f_density']))
LP_Fdensity_var_bcu = np.var(np.log10(v22LP_df_bcu['LP_f_density']))

LP_PEn_mean_bcu = np.mean(v22LP_df_bcu['Piv_En'])
LP_PEn_std_bcu = np.std(v22LP_df_bcu['Piv_En'])

# generate params for FSRQ



LPIndex_mock_dist_fsrq = np.sqrt(LP_index_var_fsrq) * np.random.randn(LP_df_mockCat_bins_fsrq) + LP_index_mean_fsrq



LPBeta_mock_dist2_fsrq = np.random.gumbel(LP_beta_mean_fsrq-0.080, np.sqrt(LP_beta_var_fsrq)-0.06, 
                                         size=LP_df_mockCat_bins_fsrq) 

LPFDensity_mock_dist_fsrq = 10**(np.sqrt(LP_Fdensity_var_fsrq) * np.random.randn(LP_df_mockCat_bins_fsrq) + LP_Fdensity_mean_fsrq) 


from scipy.stats import expon, lognorm

logshape1_fsrq, logloc1_fsrq, logscale1_fsrq = lognorm.fit(v22LP_df_fsrq['Piv_En'], loc=0)

LPPivEn_mock_dist3_fsrq = np.random.lognormal(np.log(logscale1_fsrq), logshape1_fsrq, LP_df_mockCat_bins_fsrq)
LPPivEn_mock_dist3ffsrq = LPPivEn_mock_dist3_fsrq[(LPPivEn_mock_dist3_fsrq<13000.)]


# generate params for BCU

LP_df_mockCat_bins_bcu = 11000

LPIndex_mock_dist_bcu = np.sqrt(
    LP_index_var_bcu) * np.random.randn(LP_df_mockCat_bins_bcu) + LP_index_mean_bcu


LPBeta_mock_dist2_bcu = np.random.gumbel(LP_beta_mean_bcu-0.067, np.sqrt(LP_beta_var_bcu)-0.07,
                                         size=LP_df_mockCat_bins_bcu)

LPFDensity_mock_dist_bcu = 10**(np.sqrt(LP_Fdensity_var_bcu) *
                                np.random.randn(LP_df_mockCat_bins_bcu) + LP_Fdensity_mean_bcu)


logshape1_bcu, logloc1_bcu, logscale1_bcu = lognorm.fit(
    v22LP_df_bcu['Piv_En'], loc=0)

LPPivEn_mock_dist3_bcu = np.random.lognormal(
    np.log(logscale1_bcu), logshape1_bcu, LP_df_mockCat_bins_bcu)
LPPivEn_mock_dist3fbcu = LPPivEn_mock_dist3_bcu[(
    LPPivEn_mock_dist3_bcu < 26600.)]


# mock agn LATs LONs(BLL, FSRQ, BCU)

# distribution of sinb from -1, 1
sinb_fsrq = np.random.uniform(-1, 1, LP_df_mockCat_bins_fsrq)
sinb_bcu = np.random.uniform(-1, 1, LP_df_mockCat_bins_bcu)


# b will be drawn from sinb distribution
mock_GLAT_fsrq = np.rad2deg(np.arcsin(sinb_fsrq))
mock_GLAT_bcu = np.rad2deg(np.arcsin(sinb_bcu))

# lon
gal_long_uniform_fsrq = np.random.uniform(0.0, 2*np.pi, LP_df_mockCat_bins_fsrq)
gal_long_uniform_bcu = np.random.uniform(0.0, 2*np.pi, LP_df_mockCat_bins_bcu)

mock_GLON_fsrq = np.rad2deg(gal_long_uniform_fsrq)
mock_GLON_bcu = np.rad2deg(gal_long_uniform_bcu)

#########################################
# Integration for E100 variable
#########################################


def Integrate(N, a, b, alpha, beta, PivEn,
              FluxDensity):  # number of steps: N, lower and upper limit: a, b
    value = 0
    value2 = 0

    for i in range(1, N+1):
        En1 = a + ((i-1/2) * ((b-a) / N))
        x = En1*FluxDensity * \
            ((En1/PivEn)**(- alpha - (beta * (math.log(En1/PivEn)))))
#         value += LP(En1 , alpha1, beta1, PivEn, FluxDensity)
        value += x
    value2 = ((b-a)/N) * value
    return value2

##############################################


num_cats = 5  # num_cats is number of catalog we would like to generate.
start_source_num = 0


print ('reached before for loop num cats')

for i in range(num_cats):

    # introduce the poisson noise

    noise_N_fsrq = np.random.uniform(0.7, 1.3, AGNs_fsrq_bins)
    noise_N_fsrq_comb = np.random.uniform(0.7, 1.3, AGNs_fsrq_bins+1)

    # introduce the poisson noise (used later for filling source)

    noise_N_bcu = np.random.uniform(0.7, 1.3, AGNs_bcu_bins)
    noise_N_bcu_comb = np.random.uniform(0.7, 1.3, AGNs_bcu_bins+1)

    # choose the parameters randomly from the distribution of mock parameters

    choose_number_fsrq = int(len(mock_GLON_fsrq) * random.choice(np.random.uniform(0.83, 0.92, num_cats)))
    choose_number_bcu = int(len(mock_GLON_bcu) * random.choice(np.random.uniform(0.83, 0.92, num_cats)))

    # bcu rand mock

    rand_LPIndex_mock_dist_bcu = random.sample(LPIndex_mock_dist_bcu.tolist(), choose_number_bcu)
    rand_LPBeta_mock_dist_bcu = random.sample(LPBeta_mock_dist2_bcu.tolist(), choose_number_bcu)
    rand_LPFDensity_mock_dist_bcu = random.sample(LPFDensity_mock_dist_bcu.tolist(), choose_number_bcu)
    rand_LPPivEn_mock_dist_bcu = random.sample(LPPivEn_mock_dist3_bcu.tolist(), choose_number_bcu)

    rand_GLAT_bcu = random.sample(mock_GLAT_bcu.tolist(), choose_number_bcu)
    rand_GLON_bcu = random.sample(mock_GLON_bcu.tolist(), choose_number_bcu)

    # fsrq rand mock

    rand_LPIndex_mock_dist_fsrq    = random.sample(LPIndex_mock_dist_fsrq.tolist(), choose_number_fsrq)
    rand_LPBeta_mock_dist_fsrq     = random.sample(LPBeta_mock_dist2_fsrq.tolist(), choose_number_fsrq)
    rand_LPFDensity_mock_dist_fsrq = random.sample(LPFDensity_mock_dist_fsrq.tolist(), choose_number_fsrq)
    rand_LPPivEn_mock_dist_fsrq    = random.sample(LPPivEn_mock_dist3_fsrq.tolist(), choose_number_fsrq)

    rand_GLAT_fsrq                 = random.sample(mock_GLAT_fsrq.tolist(), choose_number_fsrq)
    rand_GLON_fsrq                 = random.sample(mock_GLON_fsrq.tolist(), choose_number_fsrq)


    def simple_lum_fsrq(n, LPalpha, LPbeta, LPFd, LPPEn, LPglat, LPglon):
        result_list_fsrq = []
#       comp_list = [0] * len(E100hist_list)
        final_result_list_fsrq = []
        c_lp_a_fsrq   = [] # AGN alpha
        c_lp_b_fsrq   = [] # AGN beta
        c_lp_F_fsrq   = [] # flux density
        c_lp_PEn_fsrq = [] # pivot  energy
        c_GLAT_fsrq   = [] # lat
        c_GLON_fsrq   = [] # long
        #### loop over the number of sources (selected in random)
        for x in range(choose_number_fsrq):
            LP_index_fsrq = LPalpha[x]
            LP_beta_fsrq  = LPbeta[x]
            LP_FD_fsrq    = LPFd[x]
            LP_PEn_fsrq   = LPPEn[x]
            LP_glat_fsrq  = LPglat[x]
            LP_glon_fsrq  = LPglon[x]
# 
        # 
        # 
            result_fsrq   = Integrate(n, 100, 100e3, alpha=LP_index_fsrq, beta=LP_beta_fsrq, 
                                 PivEn=LP_PEn_fsrq, FluxDensity=LP_FD_fsrq)
            result_fsrq   = result_fsrq * 1.6e-6 
            # unit here is Erg cm^-2 s^-1:  
            if result_fsrq>=8.033498774796227e-13 and result_fsrq <= 1.1248463081262345e-09: # 10**(lbin), 10**(hbin)
                result_list_fsrq.append(result_fsrq)
# 
                c_lp_a_fsrq.append(LP_index_fsrq)
                c_lp_b_fsrq.append(LP_beta_fsrq)
                c_lp_F_fsrq.append(LP_FD_fsrq)
                c_lp_PEn_fsrq.append(LP_PEn_fsrq)
                c_GLAT_fsrq.append(LP_glat_fsrq)
                c_GLON_fsrq.append(LP_glon_fsrq)
            # 
                mockE100hist_fsrq, bins_fsrq = np.histogram(np.log10(result_list_fsrq), bins=E100bins_fsrq, density=False)
                mockE100hist_list_fsrq = mockE100hist_fsrq.tolist()
                for x1, x2, x3 in zip(mockE100hist_list_fsrq, E100hist_list_fsrq, noise_N_fsrq): 
                    if (x1*x3) > x2:
                        # print ('!!! mock higher than the real !!!', x1, x2)
                        result_list_fsrq.pop()
#                     print ('result_list len: ', len(result_list))
                        final_result_list_fsrq = result_list_fsrq[:]
                        c_lp_a_fsrq.pop()
                        c_lp_b_fsrq.pop()
                        c_lp_F_fsrq.pop()
                        c_lp_PEn_fsrq.pop()
                        c_GLAT_fsrq.pop()
                        c_GLON_fsrq.pop()
        # 
                    else:
                        continue
                        print (':) :) what is happening here :) :)')
                # 
#                 result_list.pop()    
        return final_result_list_fsrq, c_lp_a_fsrq, c_lp_b_fsrq, c_lp_F_fsrq, c_lp_PEn_fsrq, c_GLAT_fsrq, c_GLON_fsrq

    def simple_lum_bcu_comb(n, LPalpha_bcu, LPbeta_bcu, LPFd_bcu, LPPEn_bcu, LPglat_bcu, LPglon_bcu):
        result_list_bcu = []
        final_result_list_bcu = []
        c_lp_a_bcu = []  # AGN alpha
        c_lp_b_bcu = []  # AGN beta
        c_lp_F_bcu = []  # flux density
        c_lp_PEn_bcu = []  # pivot  energy
        c_GLAT_bcu = []  # lat
        c_GLON_bcu = []  # long
        # loop over the number of sources (selected in random)
        for y in range(choose_number_bcu):
            LP_index_bcu = LPalpha_bcu[y]
            LP_beta_bcu = LPbeta_bcu[y]
            LP_FD_bcu = LPFd_bcu[y]
            LP_PEn_bcu = LPPEn_bcu[y]
            LP_glat_bcu = LPglat_bcu[y]
            LP_glon_bcu = LPglon_bcu[y]

            result_bcu = Integrate(n, 100, 100e3, alpha=LP_index_bcu, beta=LP_beta_bcu,
                                   PivEn=LP_PEn_bcu, FluxDensity=LP_FD_bcu)
            result_bcu = result_bcu * 1.6e-6
            # unit here is Erg cm^-2 s^-1:
            # lowest bin -12.378695, highest bin -9.002079
            # E100hist_list_bll_combined = check_area1sigh_bll_list_pow + E100hist_list_bll_high
            # E100bins_bll_list_combined = selected_bins1_list_bll_extended + E100bins_bll_list_high
            # 10**(lbin), 10**(hbin)
            if result_bcu >= 10**E100bins_bcu_list_combined[0] and result_bcu < 10**(E100bins_bcu_list_combined[-1]):
                result_list_bcu.append(result_bcu)

                c_lp_a_bcu.append(LP_index_bcu)
                c_lp_b_bcu.append(LP_beta_bcu)
                c_lp_F_bcu.append(LP_FD_bcu)
                c_lp_PEn_bcu.append(LP_PEn_bcu)
                c_GLAT_bcu.append(LP_glat_bcu)
                c_GLON_bcu.append(LP_glon_bcu)

                mockE100hist_bcu, bins_bcu = np.histogram(
                    np.log10(result_list_bcu), bins=E100bins_bcu_list_combined, density=False)
                mockE100hist_list_bcu = mockE100hist_bcu.tolist()
#               for Ebins in E100bins_notlast:
                for x1bcu, x2bcu, x3bcu in zip(mockE100hist_list_bcu, E100hist_list_bcu_combined, noise_N_bcu_comb):
                    if (x1bcu*x3bcu) > x2bcu:
                        result_list_bcu.pop()
#                     print ('result_list len: ', len(result_list))
                        final_result_list_bcu = result_list_bcu[:]
                        c_lp_a_bcu.pop()
                        c_lp_b_bcu.pop()
                        c_lp_F_bcu.pop()
                        c_lp_PEn_bcu.pop()
                        c_GLAT_bcu.pop()
                        c_GLON_bcu.pop()

                    else:
                        continue
                        print(':) :) what is happening here :) :)')

#                   result_list.pop()
        return final_result_list_bcu, c_lp_a_bcu, c_lp_b_bcu, c_lp_F_bcu, c_lp_PEn_bcu, c_GLAT_bcu, c_GLON_bcu

    final_result_list_bcu_comb, c_lp_a_bcu_comb, c_lp_b_bcu_comb, c_lp_F_bcu_comb, c_lp_PEn_bcu_comb, c_GLAT_bcu_comb, c_GLON_bcu_comb = simple_lum_bcu_comb(1070,
                                                                                                                                                             rand_LPIndex_mock_dist_bcu,
                                                                                                                                                             rand_LPBeta_mock_dist_bcu, 
                                                                                                                                                             rand_LPFDensity_mock_dist_bcu, rand_LPPivEn_mock_dist_bcu, 
                                                                                                                                                             rand_GLAT_bcu, rand_GLON_bcu)

    final_result_list_fsrq, c_lp_a_fsrq, c_lp_b_fsrq, c_lp_F_fsrq, c_lp_PEn_fsrq, c_GLAT_fsrq, c_GLON_fsrq = simple_lum_fsrq(1070, rand_LPIndex_mock_dist_fsrq, rand_LPBeta_mock_dist_fsrq, rand_LPFDensity_mock_dist_fsrq, 
                                                                                                                            rand_LPPivEn_mock_dist_fsrq, 
                                                                                                                            rand_GLAT_fsrq, rand_GLON_fsrq)

    ### randomly select 35% of entries from BCU, this will be for bll 
    ### randomly select 28% of entries from BCU, this will be for fsrq

    c_lp_a_bcu_comb_rand_samp = random.sample(c_lp_a_bcu_comb, int(len(c_lp_a_bcu_comb)*0.62))
    c_lp_a_bcu_comb_bll = c_lp_a_bcu_comb_rand_samp[0: int(len(c_lp_a_bcu_comb)*0.35)]
    c_lp_a_bcu_comb_fsrq = c_lp_a_bcu_comb_rand_samp[int(len(c_lp_a_bcu_comb)*0.35):]


    c_lp_b_bcu_comb_rand_samp = random.sample(c_lp_b_bcu_comb, int(len(c_lp_b_bcu_comb)*0.62))
    c_lp_b_bcu_comb_bll = c_lp_b_bcu_comb_rand_samp[0: int(len(c_lp_b_bcu_comb)*0.35)]
    c_lp_b_bcu_comb_fsrq = c_lp_b_bcu_comb_rand_samp[int(len(c_lp_b_bcu_comb)*0.35):]

    c_lp_F_bcu_comb_rand_samp = random.sample(c_lp_F_bcu_comb, int(len(c_lp_F_bcu_comb)*0.62))
    c_lp_F_bcu_comb_bll = c_lp_F_bcu_comb_rand_samp[0: int(len(c_lp_F_bcu_comb)*0.35)]
    c_lp_F_bcu_comb_fsrq = c_lp_F_bcu_comb_rand_samp[int(len(c_lp_F_bcu_comb)*0.35):]

    c_lp_PEn_bcu_comb_rand_samp = random.sample(c_lp_PEn_bcu_comb, int(len(c_lp_PEn_bcu_comb)*0.62))
    c_lp_PEn_bcu_comb_bll = c_lp_PEn_bcu_comb_rand_samp[0: int(len(c_lp_PEn_bcu_comb)*0.35)]
    c_lp_PEn_bcu_comb_fsrq = c_lp_PEn_bcu_comb_rand_samp[int(len(c_lp_PEn_bcu_comb)*0.35):]

    c_GLON_bcu_comb_rand_samp = random.sample(c_GLON_bcu_comb, int(len(c_GLON_bcu_comb)*0.62))
    c_GLON_bcu_comb_bll = c_GLON_bcu_comb_rand_samp[0: int(len(c_GLON_bcu_comb)*0.35)]
    c_GLON_bcu_comb_fsrq = c_GLON_bcu_comb_rand_samp[int(len(c_GLON_bcu_comb)*0.35):]

    c_GLAT_bcu_comb_rand_samp = random.sample(c_GLAT_bcu_comb, int(len(c_GLAT_bcu_comb)*0.62))
    c_GLAT_bcu_comb_bll = c_GLAT_bcu_comb_rand_samp[0: int(len(c_GLAT_bcu_comb)*0.35)]
    c_GLAT_bcu_comb_fsrq = c_GLAT_bcu_comb_rand_samp[int(len(c_GLAT_bcu_comb)*0.35):]

 

    # c_lp_a_bll_comb_bcu = c_lp_a_bll + c_lp_a_bcu_comb_bll
    # c_lp_b_bll_comb_bcu = c_lp_b_bll + c_lp_b_bcu_comb_bll
    # c_lp_PEn_bll_comb_bcu = c_lp_PEn_bll + c_lp_PEn_bcu_comb_bll
    # c_lp_F_bll_comb_bcu = c_lp_F_bll + c_lp_F_bcu_comb_bll
    # c_GLON_bll_comb_bcu = c_GLON_bll + c_GLON_bcu_comb_bll
    # c_GLAT_bll_comb_bcu = c_GLAT_bll + c_GLAT_bcu_comb_bll

    c_lp_a_fsrq_comb_bcu = c_lp_a_fsrq + c_lp_b_bcu_comb_fsrq
    c_lp_b_fsrq_comb_bcu = c_lp_b_fsrq + c_lp_F_bcu_comb_fsrq
    c_lp_F_fsrq_comb_bcu = c_lp_F_fsrq + c_lp_F_bcu_comb_fsrq
    c_lp_PEn_fsrq_comb_bcu = c_lp_PEn_fsrq + c_lp_PEn_bcu_comb_fsrq
    c_GLON_fsrq_comb_bcu = c_GLON_fsrq + c_GLON_bcu_comb_fsrq
    c_GLAT_fsrq_comb_bcu = c_GLAT_fsrq + c_GLAT_bcu_comb_fsrq

    c_ra_cord_list_fsrq_bcu = []
    c_dec_cord_list_fsrq_bcu = []

    for clo, cla in zip(c_GLON_fsrq_comb_bcu, c_GLAT_fsrq_comb_bcu):
        c_icrs1_c  = SkyCoord(l=clo*u.degree, b=cla*u.degree, frame='galactic')
        c_ra_cord_list_fsrq_bcu.append(c_icrs1_c.fk5.ra.degree)
        c_dec_cord_list_fsrq_bcu.append(c_icrs1_c.fk5.dec.degree)

    mock_source_num_fsrq_comb_bcu = [i for i in range(len(c_lp_a_fsrq_comb_bcu))]

    #### here we randomly select the variability and multiply with pre-factor 
    #### 

    year_mean_mock_dist3ffsrq_list_new_select = random.sample(year_mean_mock_dist3ffsrq_list_new, 
                                                         len(c_lp_F_fsrq_comb_bcu))


    print ('check sampling and org K length: ', len(year_mean_mock_dist3ffsrq_list_new_select), len(c_lp_F_fsrq_comb_bcu))

    variability_fsrq_bcu_F = [a*b for a, b in zip(year_mean_mock_dist3ffsrq_list_new_select, c_lp_F_fsrq_comb_bcu)]
    # variability_fsrq_bcu_F = [a for a in ]

    if (len(c_lp_a_fsrq_comb_bcu) <= 1370) and (len(c_lp_a_fsrq_comb_bcu) >= 980):
        source_numbers_check = len(c_lp_a_fsrq_comb_bcu)
        print ('sources in file: ', source_numbers_check)
        mock_xmlfile_fsrq_bcu_comb = open('mock_4FGL_V22_fsrqvar_comb_bcu_file%dS%d.xml' %(start_source_num, source_numbers_check), 'w')
        mock_xmlfile_fsrq_bcu_comb.write('<source_library title="source library">\n')
        for n, al, be, pivE, ra, dec, num in zip(variability_fsrq_bcu_F, c_lp_a_fsrq_comb_bcu, c_lp_b_fsrq_comb_bcu, 
                                         c_lp_PEn_fsrq_comb_bcu, 
                                         c_ra_cord_list_fsrq_bcu, c_dec_cord_list_fsrq_bcu, mock_source_num_fsrq_comb_bcu):

            if n > 0.0:
                mock_xmlfile_fsrq_bcu_comb.write('<source name="LogParabola_source{0}" type="PointSource">\n'.format(num))
                mock_xmlfile_fsrq_bcu_comb.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
                mock_xmlfile_fsrq_bcu_comb.write('<spectrum type="LogParabola">\n')
                mock_xmlfile_fsrq_bcu_comb.write(
                    '<parameter free="1" max="{0}" min="{1}" name="norm" scale="1.0" value="{2}"/>\n'.format(max(variability_fsrq_bcu_F), min(variability_fsrq_bcu_F), n))
                mock_xmlfile_fsrq_bcu_comb.write(
                    '<parameter free="1" max="{0}" min="{1}" name="alpha" scale="1.0" value="{2}"/>\n'.format(max(c_lp_a_fsrq_comb_bcu), min(c_lp_a_fsrq_comb_bcu), al))
                mock_xmlfile_fsrq_bcu_comb.write(
                    '<parameter free="1" max="{0}" min="{1}" name="Eb" scale="1" value="{2}"/>\n'.format(max(c_lp_PEn_fsrq_comb_bcu), min(c_lp_PEn_fsrq_comb_bcu), pivE))
                mock_xmlfile_fsrq_bcu_comb.write(
                    '<parameter free="1" max="{0}" min="{1}" name="beta" scale="1.0" value="{2}"/>\n'.format(max(c_lp_b_fsrq_comb_bcu), min(c_lp_b_fsrq_comb_bcu), be))
                mock_xmlfile_fsrq_bcu_comb.write('</spectrum>\n')
                mock_xmlfile_fsrq_bcu_comb.write('<spatialModel type="SkyDirFunction">\n')
                mock_xmlfile_fsrq_bcu_comb.write(
                    '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
                mock_xmlfile_fsrq_bcu_comb.write(
                    '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
                mock_xmlfile_fsrq_bcu_comb.write('</spatialModel>\n')
                mock_xmlfile_fsrq_bcu_comb.write('</source>\n')
    
        mock_xmlfile_fsrq_bcu_comb.write('</source_library>')    
        mock_xmlfile_fsrq_bcu_comb.close() 

    else:
        continue
        print ('!!! shouldnt reach here !!!')    
    start_source_num = start_source_num + 1    
