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

v22LP_df_bll = v22LP_df_AGNs_Flux_History[(v22LP_df_AGNs_Flux_History['Class1'] == 'bll  ') | (
    v22LP_df_AGNs_Flux_History.Class1 == 'BLL  ')]

v22LP_df_bcu = v22LP_df_AGNs_Flux_History[(v22LP_df_AGNs_Flux_History['Class1'] == 'bcu  ') | (
    v22LP_df_AGNs_Flux_History.Class1 == 'BCU  ')]

AGNs_bll_bins = int(np.sqrt(v22LP_df_bll.shape[0]))
AGNs_bcu_bins = int(np.sqrt(v22LP_df_bcu.shape[0]))


# E100 for blls

E100hist_bll, E100bins_bll, _ = plt.hist(np.log10(v22LP_df_bll['En_flux_100']), bins=AGNs_bll_bins,
                                         density=False, alpha=0.8, edgecolor='navy', color='lime')

plt.clf()

E100hist_list_bll = E100hist_bll.tolist()


selected_hist1_bll_fit = E100hist_bll[2:6]
selected_hist1_bll = E100hist_bll[0:6]
selected_hist1_list_bll_fit = [np.log10(k) for k in selected_hist1_bll_fit]


E100bins_bll_list = E100bins_bll.tolist()
selected_bins1_bll_fit = E100bins_bll[2:6]
selected_bins1_bll = E100bins_bll[0:6]


selected_bins1_list_bll_fit = [i for i in selected_bins1_bll_fit]
selected_bins1_list_bll = [i for i in selected_bins1_bll]
selected_bins1_list_bll_extended = [
    selected_bins1_list_bll[0]-selected_bins1_list_bll[1] + selected_bins1_list_bll[0]] + selected_bins1_list_bll

# print ('check extended bins bll list: ', selected_bins1_list_bll_extended)


reg_bll = LinearRegression(fit_intercept=True)
reg_bll.fit(np.reshape(selected_bins1_list_bll_fit,
                       (-1, 1)), selected_hist1_list_bll_fit)

# # reg.fit(selected_bins1_list, selected_hist1_list)
# print ('check fit values: intercept: ', reg_bll.intercept_)
# print ('check fit values: coeff and intercept: ', reg_bll.coef_[0], reg_bll.intercept_)

hist_vals_bll_check = [(reg_bll.coef_[0] * i + reg_bll.intercept_)
                       for i in selected_bins1_list_bll_fit]
hist_vals_bll_check_extended = [
    (reg_bll.coef_[0] * i + reg_bll.intercept_) for i in selected_bins1_list_bll_extended]
selected_bins1_bll_extended_arr = np.array(selected_bins1_list_bll_extended)

y_err1sig_bll = selected_bins1_bll_extended_arr.std() * np.sqrt(1/len(selected_bins1_bll_extended_arr) + (selected_bins1_bll_extended_arr -
                                                                                                          selected_bins1_bll_extended_arr.mean())**2 / np.sum((selected_bins1_bll_extended_arr - selected_bins1_bll_extended_arr.mean())**2))
y_err2sig_bll = (2*selected_bins1_bll_extended_arr.std()) * np.sqrt(1/len(selected_bins1_bll_extended_arr) + (selected_bins1_bll_extended_arr -
                                                                                                              selected_bins1_bll_extended_arr.mean())**2 / np.sum((selected_bins1_bll_extended_arr - selected_bins1_bll_extended_arr.mean())**2))

# # hist_vals = [reg.coef_ * i + reg.intercept_ for i in selected_bins1_list]
# print ('fitted hist vals: ', hist_vals_bll)


check_area1sigl_bll = hist_vals_bll_check_extended - y_err1sig_bll
check_area1sigh_bll = hist_vals_bll_check_extended + y_err1sig_bll
check_area2sigh_bll = hist_vals_bll_check_extended + y_err2sig_bll

# # ax1.fill_between(selected_bins1_list_bll, np.power(10, check_area1), np.power(10, check_area2), alpha=0.2)


check_area1sigh_bll_list = check_area1sigh_bll.tolist()
check_area1sigh_bll_list_pow = [10**i for i in check_area1sigh_bll_list]

E100hist_list_bll_high = E100hist_list_bll[6:]
E100bins_bll_list_high = E100bins_bll_list[6:]


E100hist_list_bll_combined = check_area1sigh_bll_list_pow + E100hist_list_bll_high
E100bins_bll_list_combined = selected_bins1_list_bll_extended + E100bins_bll_list_high

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


LP_index_mean_bll = v22LP_df_bll['LP_index'].mean()
LP_index_var_bll = v22LP_df_bll['LP_index'].var()

LP_beta_mean_bll = v22LP_df_bll['LP_beta'].mean()
LP_beta_var_bll = v22LP_df_bll['LP_beta'].var()

# for conversion 10**(LP_Fdensity_mean)
LP_Fdensity_mean_bll = np.mean(np.log10(v22LP_df_bll['LP_f_density']))
LP_Fdensity_var_bll = np.var(np.log10(v22LP_df_bll['LP_f_density']))

LP_PEn_mean_bll = np.mean(v22LP_df_bll['Piv_En'])
LP_PEn_std_bll = np.std(v22LP_df_bll['Piv_En'])


LP_index_mean_bcu = v22LP_df_bcu['LP_index'].mean()
LP_index_var_bcu = v22LP_df_bcu['LP_index'].var()

LP_beta_mean_bcu = v22LP_df_bcu['LP_beta'].mean()
LP_beta_var_bcu = v22LP_df_bcu['LP_beta'].var()

# for conversion 10**(LP_Fdensity_mean)
LP_Fdensity_mean_bcu = np.mean(np.log10(v22LP_df_bcu['LP_f_density']))
LP_Fdensity_var_bcu = np.var(np.log10(v22LP_df_bcu['LP_f_density']))

LP_PEn_mean_bcu = np.mean(v22LP_df_bcu['Piv_En'])
LP_PEn_std_bcu = np.std(v22LP_df_bcu['Piv_En'])

# generate params for BLL

LP_df_mockCat_bins_bll = 10000

LPIndex_mock_dist_bll = np.sqrt(
    LP_index_var_bll) * np.random.randn(LP_df_mockCat_bins_bll) + LP_index_mean_bll


LPBeta_mock_dist2_bll = np.random.gumbel(LP_beta_mean_bll-0.067, np.sqrt(LP_beta_var_bll)-0.08,
                                         size=LP_df_mockCat_bins_bll)
# best for beta

LPFDensity_mock_dist_bll = 10**(np.sqrt(LP_Fdensity_var_bll) *
                                np.random.randn(LP_df_mockCat_bins_bll) + LP_Fdensity_mean_bll)


logshape1_bll, logloc1_bll, logscale1_bll = lognorm.fit(
    v22LP_df_bll['Piv_En'], loc=0)

LPPivEn_mock_dist3_bll = np.random.lognormal(
    np.log(logscale1_bll), logshape1_bll, LP_df_mockCat_bins_bll)
LPPivEn_mock_dist3fbll = LPPivEn_mock_dist3_bll[(
    LPPivEn_mock_dist3_bll < 26000.)]


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
sinb_bll = np.random.uniform(-1, 1, LP_df_mockCat_bins_bll)
sinb_bcu = np.random.uniform(-1, 1, LP_df_mockCat_bins_bcu)


# b will be drawn from sinb distribution
mock_GLAT_bll = np.rad2deg(np.arcsin(sinb_bll))
mock_GLAT_bcu = np.rad2deg(np.arcsin(sinb_bcu))

# lon
gal_long_uniform_bll = np.random.uniform(
    0.0, 2*np.pi, LP_df_mockCat_bins_bll)  # already in radian
gal_long_uniform_bcu = np.random.uniform(0.0, 2*np.pi, LP_df_mockCat_bins_bcu)

mock_GLON_bll = np.rad2deg(gal_long_uniform_bll)
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

    noise_N_bll_comb = np.random.uniform(0.7, 1.3, AGNs_bll_bins+1)
    noise_N_bll = np.random.uniform(0.7, 1.3, AGNs_bll_bins)

    # introduce the poisson noise (used later for filling source)

    noise_N_bcu = np.random.uniform(0.7, 1.3, AGNs_bcu_bins)
    noise_N_bcu_comb = np.random.uniform(0.7, 1.3, AGNs_bcu_bins+1)

    # choose the parameters randomly from the distribution of mock parameters

    choose_number_bll = int(len(mock_GLON_bll) * random.choice(np.random.uniform(0.83, 0.93, num_cats)))
    choose_number_bcu = int(len(mock_GLON_bcu) * random.choice(np.random.uniform(0.83, 0.93, num_cats)))

    # bcu rand mock

    rand_LPIndex_mock_dist_bcu = random.sample(LPIndex_mock_dist_bcu.tolist(), choose_number_bcu)
    rand_LPBeta_mock_dist_bcu = random.sample(LPBeta_mock_dist2_bcu.tolist(), choose_number_bcu)
    rand_LPFDensity_mock_dist_bcu = random.sample(LPFDensity_mock_dist_bcu.tolist(), choose_number_bcu)
    rand_LPPivEn_mock_dist_bcu = random.sample(LPPivEn_mock_dist3_bcu.tolist(), choose_number_bcu)

    rand_GLAT_bcu = random.sample(mock_GLAT_bcu.tolist(), choose_number_bcu)
    rand_GLON_bcu = random.sample(mock_GLON_bcu.tolist(), choose_number_bcu)

    # bll rand mock

    rand_LPIndex_mock_dist_bll = random.sample(LPIndex_mock_dist_bll.tolist(), choose_number_bll)
    rand_LPBeta_mock_dist_bll = random.sample(LPBeta_mock_dist2_bll.tolist(), choose_number_bll)
    rand_LPFDensity_mock_dist_bll = random.sample(LPFDensity_mock_dist_bll.tolist(), choose_number_bll)
    rand_LPPivEn_mock_dist_bll = random.sample(LPPivEn_mock_dist3_bll.tolist(), choose_number_bll)

    rand_GLAT_bll = random.sample(mock_GLAT_bll.tolist(), choose_number_bll)
    rand_GLON_bll = random.sample(mock_GLON_bll.tolist(), choose_number_bll)

    def simple_lum_bll(n, LPalpha, LPbeta, LPFd, LPPEn, LPglat, LPglon):
        result_list_bll = []
        final_result_list_bll = []
        c_lp_a_bll = []  # AGN alpha
        c_lp_b_bll = []  # AGN beta
        c_lp_F_bll = []  # flux density
        c_lp_PEn_bll = []  # pivot  energy
        c_GLAT_bll = []  # lat
        c_GLON_bll = []  # long
    # loop over the number of sources (selected in random)
        for x in range(choose_number_bll):
            LP_index_bll = LPalpha[x]
            LP_beta_bll = LPbeta[x]
            LP_FD_bll = LPFd[x]
            LP_PEn_bll = LPPEn[x]
            LP_glat_bll = LPglat[x]
            LP_glon_bll = LPglon[x]

            result_bll = Integrate(n, 100, 100e3, alpha=LP_index_bll, beta=LP_beta_bll,
                                   PivEn=LP_PEn_bll, FluxDensity=LP_FD_bll)
            # n is the number of steps in integration, higher ---> better accuracy
            # unit here is MeV cm^-2 s^-1 :
            result_bll = result_bll * 1.6e-6
            # unit here is Erg cm^-2 s^-1:
            # lowest bin -12.378695, highest bin -9.002079
            # 10**(lbin), 10**(hbin)
            if result_bll >= 7.18408151614367978e-13 and result_bll <= 4.4967794722950834e-10:
                result_list_bll.append(result_bll)

                c_lp_a_bll.append(LP_index_bll)
                c_lp_b_bll.append(LP_beta_bll)
                c_lp_F_bll.append(LP_FD_bll)
                c_lp_PEn_bll.append(LP_PEn_bll)
                c_GLAT_bll.append(LP_glat_bll)
                c_GLON_bll.append(LP_glon_bll)

                mockE100hist_bll, bins_bll = np.histogram(
                    np.log10(result_list_bll), bins=E100bins_bll, density=False)
                mockE100hist_list_bll = mockE100hist_bll.tolist()
#               for Ebins in E100bins_notlast:
                for x1, x2, x3 in zip(mockE100hist_list_bll, E100hist_list_bll, noise_N_bll):
                    if (x1*x3) > x2:
                        result_list_bll.pop()
                        final_result_list_bll = result_list_bll[:]
                        c_lp_a_bll.pop()
                        c_lp_b_bll.pop()
                        c_lp_F_bll.pop()
                        c_lp_PEn_bll.pop()
                        c_GLAT_bll.pop()
                        c_GLON_bll.pop()

                    else:
                        continue
                        print(':) :) what is happening here :) :)')

#                 result_list.pop()
        return final_result_list_bll, c_lp_a_bll, c_lp_b_bll, c_lp_F_bll, c_lp_PEn_bll, c_GLAT_bll, c_GLON_bll

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

    final_result_list_bll, c_lp_a_bll, c_lp_b_bll, c_lp_F_bll, c_lp_PEn_bll, c_GLAT_bll, c_GLON_bll = simple_lum_bll(1070,
                                                                                                                     rand_LPIndex_mock_dist_bll,
                                                                                                                     rand_LPBeta_mock_dist_bll,
                                                                                                                     rand_LPFDensity_mock_dist_bll,
                                                                                                                     rand_LPPivEn_mock_dist_bll,
                                                                                                                     rand_GLAT_bll, rand_GLON_bll)

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

 

    c_lp_a_bll_comb_bcu = c_lp_a_bll + c_lp_a_bcu_comb_bll
    c_lp_b_bll_comb_bcu = c_lp_b_bll + c_lp_b_bcu_comb_bll
    c_lp_PEn_bll_comb_bcu = c_lp_PEn_bll + c_lp_PEn_bcu_comb_bll
    c_lp_F_bll_comb_bcu = c_lp_F_bll + c_lp_F_bcu_comb_bll
    c_GLON_bll_comb_bcu = c_GLON_bll + c_GLON_bcu_comb_bll
    c_GLAT_bll_comb_bcu = c_GLAT_bll + c_GLAT_bcu_comb_bll

    # c_lp_a_fsrq_comb_bcu = c_lp_a_fsrq + c_lp_b_bcu_comb_fsrq
    # c_lp_b_fsrq_comb_bcu = c_lp_b_fsrq + c_lp_F_bcu_comb_fsrq
    # c_lp_F_fsrq_comb_bcu = c_lp_F_fsrq + c_lp_F_bcu_comb_fsrq
    # c_lp_PEn_fsrq_comb_bcu = c_lp_PEn_fsrq + c_lp_PEn_bcu_comb_fsrq
    # c_GLON_fsrq_comb_bcu = c_GLON_fsrq + c_GLON_bcu_comb_fsrq
    # c_GLAT_fsrq_comb_bcu = c_GLAT_fsrq + c_GLAT_bcu_comb_fsrq

    c_ra_cord_list_bll_bcu = []
    c_dec_cord_list_bll_bcu = []

    for clo, cla in zip(c_GLON_bll_comb_bcu, c_GLAT_bll_comb_bcu):
        c_icrs1_c  = SkyCoord(l=clo*u.degree, b=cla*u.degree, frame='galactic')
        c_ra_cord_list_bll_bcu.append(c_icrs1_c.fk5.ra.degree)
        c_dec_cord_list_bll_bcu.append(c_icrs1_c.fk5.dec.degree)

    mock_source_num_bll_comb_bcu = [i for i in range(len(c_lp_a_bll_comb_bcu))]

    if (len(c_lp_a_bll_comb_bcu) <= 1950) and (len(c_lp_a_bll_comb_bcu) >= 1300):
        source_numbers_check = len(c_lp_a_bll_comb_bcu)
        print ('sources in file: ', source_numbers_check)
        mock_xmlfile_bll_bcu_comb = open('./gen_bll/mock_4FGL_V22_bll_comb_bcu_file%dS%d.xml' %(start_source_num, source_numbers_check), 'w')
        mock_xmlfile_bll_bcu_comb.write('<source_library title="source library">\n')
        for n, al, be, pivE, ra, dec, num in zip(c_lp_F_bll_comb_bcu, c_lp_a_bll_comb_bcu, c_lp_b_bll_comb_bcu, 
                                         c_lp_PEn_bll_comb_bcu, 
                                         c_ra_cord_list_bll_bcu, c_dec_cord_list_bll_bcu, mock_source_num_bll_comb_bcu):
    
            mock_xmlfile_bll_bcu_comb.write('<source name="LogParabola_source{0}" type="PointSource">\n'.format(num))
            mock_xmlfile_bll_bcu_comb.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
            mock_xmlfile_bll_bcu_comb.write('<spectrum type="LogParabola">\n')
            mock_xmlfile_bll_bcu_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="norm" scale="1.0" value="{2}"/>\n'.format(max(c_lp_F_bll_comb_bcu), min(c_lp_F_bll_comb_bcu), n))
            mock_xmlfile_bll_bcu_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="alpha" scale="1.0" value="{2}"/>\n'.format(max(c_lp_a_bll_comb_bcu), min(c_lp_a_bll_comb_bcu), al))
            mock_xmlfile_bll_bcu_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="Eb" scale="1" value="{2}"/>\n'.format(max(c_lp_PEn_bll_comb_bcu), min(c_lp_PEn_bll_comb_bcu), pivE))
            mock_xmlfile_bll_bcu_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="beta" scale="1.0" value="{2}"/>\n'.format(max(c_lp_b_bll_comb_bcu), min(c_lp_b_bll_comb_bcu), be))
            mock_xmlfile_bll_bcu_comb.write('</spectrum>\n')
            mock_xmlfile_bll_bcu_comb.write('<spatialModel type="SkyDirFunction">\n')
            mock_xmlfile_bll_bcu_comb.write(
                '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
            mock_xmlfile_bll_bcu_comb.write(
                '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
            mock_xmlfile_bll_bcu_comb.write('</spatialModel>\n')
            mock_xmlfile_bll_bcu_comb.write('</source>\n')
    
        mock_xmlfile_bll_bcu_comb.write('</source_library>')    
        mock_xmlfile_bll_bcu_comb.close() 

    else:
        continue
        print ('!!! shouldnt reach here !!!')    
    start_source_num = start_source_num + 1 
