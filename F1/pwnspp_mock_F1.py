import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import expon, lognorm
from scipy.optimize import curve_fit

from astropy.io import fits


from astropy import units as u
from astropy.coordinates import SkyCoord

###############################################################

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

# select pwn_spp_snr

v22LP_df_pwn_spp_snr = v22LP_df_flux_history[(v22LP_df_flux_history['Class1']=='pwn  ')|(v22LP_df_flux_history.Class1=='PWN  ') | (v22LP_df_flux_history.Class1=='spp  ')|  (v22LP_df_flux_history.Class1=='snr  ')|  (v22LP_df_flux_history.Class1=='SNR  ')]
# print (v22LP_df_pwn_spp_snr.shape)

AGNs_spp_pwn_snr_bins = 15 #int(np.sqrt(v22LP_df_pwn_spp_snr.shape[0]))

LP_index_mean_pwn_spp_snr = v22LP_df_pwn_spp_snr['LP_index'].mean()
LP_index_var_pwn_spp_snr = v22LP_df_pwn_spp_snr['LP_index'].var()

LP_beta_mean_pwn_spp_snr = v22LP_df_pwn_spp_snr['LP_beta'].mean()
LP_beta_var_pwn_spp_snr = v22LP_df_pwn_spp_snr['LP_beta'].var()

LP_Fdensity_mean_pwn_spp_snr = np.mean(np.log10(v22LP_df_pwn_spp_snr['LP_f_density'])) # for conversion 10**(LP_Fdensity_mean)
LP_Fdensity_var_pwn_spp_snr = np.var(np.log10(v22LP_df_pwn_spp_snr['LP_f_density']))

LP_PEn_mean_pwn_spp_snr = np.mean(v22LP_df_pwn_spp_snr['Piv_En']) 
LP_PEn_std_pwn_spp_snr = np.std(v22LP_df_pwn_spp_snr['Piv_En'])

#### prepare mock pwn_spp_snr

LP_df_mockCat_bins_pwn_spp_snr = 3000

LPIndex_mock_dist_pwn_spp_snr = np.sqrt(LP_index_var_pwn_spp_snr) * np.random.randn(LP_df_mockCat_bins_pwn_spp_snr) + LP_index_mean_pwn_spp_snr



LPBeta_mock_dist2_pwn_spp_snr = np.random.gumbel(LP_beta_mean_pwn_spp_snr-0.067, np.sqrt(LP_beta_var_pwn_spp_snr)-0.07, 
                                         size=LP_df_mockCat_bins_pwn_spp_snr) 
# best for beta

LPFDensity_mock_dist_pwn_spp_snr = 10**(np.sqrt(LP_Fdensity_var_pwn_spp_snr) * np.random.randn(LP_df_mockCat_bins_pwn_spp_snr) + LP_Fdensity_mean_pwn_spp_snr)



logshape1_pwn_spp_snr, logloc1_pwn_spp_snr, logscale1_pwn_spp_snr = lognorm.fit(v22LP_df_pwn_spp_snr['Piv_En'], loc=0.001)

# print ('check fit shape, loc and scale, log(scale): ', logshape1_pwn_spp_snr, logloc1_pwn_spp_snr, logscale1_pwn_spp_snr, 
#        np.log(logscale1_pwn_spp_snr))

LPPivEn_mock_dist3_pwn_spp_snr = np.random.lognormal(np.log(logscale1_pwn_spp_snr), logshape1_pwn_spp_snr, 
                                                     LP_df_mockCat_bins_pwn_spp_snr)

LPPivEn_mock_dist3fpwn_spp_snr = LPPivEn_mock_dist3_pwn_spp_snr[(LPPivEn_mock_dist3_pwn_spp_snr<239424.859375)]

### LP Parametrization and Integrated Flux 

def Integrate(N, a, b, alpha, beta, PivEn, 
              FluxDensity): # number of steps, lower and upper limit 
    value = 0
    value2 = 0
    
    for i in range(1, N+1):
        En1 = a + ( (i-1/2)* ( (b-a)/ N ) )
        x = En1*FluxDensity * ((En1/PivEn)**(- alpha - ( beta *(math.log(En1/PivEn)) ) ))
#         value += LP(En1 , alpha1, beta1, PivEn, FluxDensity)
        value += x
    value2 = ( (b-a)/N ) * value
    return value2


### LAT and LON (Distribution Similar to PSRs)

pwn_spp_GLON_arr = v22LP_df_pwn_spp_snr[['GLON']].to_numpy()
pwn_spp_GLAT_arr = v22LP_df_pwn_spp_snr[['GLAT']].to_numpy()
# print (pwn_spp_GLON_arr.shape)
pwn_spp_GLON_arr_where = np.where(pwn_spp_GLON_arr <= 180., pwn_spp_GLON_arr, pwn_spp_GLON_arr-360.)

pwn_spp_latcoord = sorted(np.random.uniform(-45, 30, 300)) # from uniform but sorted (alternative np.linspace)



def gauss2mix(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))


npwnLAT, pwnbinsLAT, _ = plt.hist(v22LP_df_pwn_spp_snr['GLAT'], bins=AGNs_spp_pwn_snr_bins, 
                                  label='histogram', alpha=0.7, density=True)

plt.clf()

bin_centerpwnLAT = (pwnbinsLAT[:-1] + pwnbinsLAT[1:])/2

# print ('bin counts: ', npwnLAT)
# print ('bins: ', pwnbinsLAT)

p0lat_pwn = [0.22, -0.05, 3, 0.04, -0.05, 15]
params_pwn, params_cov_pwn = curve_fit(gauss2mix, bin_centerpwnLAT, npwnLAT, p0=p0lat_pwn)

# print ('check fitted params 1st and 2nd gaussian : ', params_pwn, params_pwn[0:3], 
#        params_pwn[3:])

def firstgauss(x, p1):
    a1, mu1, sigma1 = p1
    return a1*np.exp(-(x-mu1)**2/(2.*sigma1**2))
def secondgauss(x, p2):
    a2, mu2, sigma2 = p2
    return a2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

# possible_hist_lat = [i*j for i, j in zip(pleclatcoord, his_fitPLECLAT)]

# ### get the fitted curve 
his_fitpwnLAT = gauss2mix(pwn_spp_latcoord, *params_pwn)
first_hist_pwn_spp = firstgauss(pwn_spp_latcoord, params_pwn[0:3])
second_hist_pwn_spp = secondgauss(pwn_spp_latcoord, params_pwn[3:])



pwn_spp_loncoord = sorted(np.random.uniform(-180, 180, 300))


npwnLON, pwnbinsLON, _ = plt.hist(pwn_spp_GLON_arr_where, bins=AGNs_spp_pwn_snr_bins, label='histogram', alpha=0.7, 
                            density=True)


plt.clf()

bin_centerpwnLON = (pwnbinsLON[:-1] + pwnbinsLON[1:])/2


def gaussoffsetLON(x, *p):
    A1, mu1, sigma1 = p
    g1 = A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) 
    return g1 

p0lon_pwn = [2e-2, 0, 70]

paramsLON_pwn, params_covLON_pwn = curve_fit(gaussoffsetLON, bin_centerpwnLON, npwnLON, p0=p0lon_pwn)



def firstgaussoffset(x, p1):
    a1, mu1, sigma1 = p1
    return a1*np.exp(-(x-mu1)**2/(2.*sigma1**2))


his_fitpwnLON = gaussoffsetLON(pwn_spp_loncoord, *paramsLON_pwn)


possible_pwn_spp_lat1 = np.random.normal(-0.05,  params_pwn[2], 800)

possible_pwn_spp_lat1_selected = np.random.choice(possible_pwn_spp_lat1, 700)

possible_pwn_spp_lat2 = np.random.normal(-0.5, params_pwn[5], 800)

possible_pwn_spp_lat2_selected = np.random.choice(possible_pwn_spp_lat2, 700)

total_hist_pwn_spp = (possible_pwn_spp_lat1) + (possible_pwn_spp_lat2)
total_histLAT1_pwn_spp = np.concatenate((possible_pwn_spp_lat1_selected, possible_pwn_spp_lat2_selected))

total_histLAT1f_pwn_spp = total_histLAT1_pwn_spp[(total_histLAT1_pwn_spp>-50.) & (total_histLAT1_pwn_spp<30.)]
# print ('check min and max of LAT hist after selection: ', max(total_histLAT1f_pwn_spp), min(total_histLAT1f_pwn_spp))
# print ('selected number of LATs: ', len(total_histLAT1f_pwn_spp))





mock_num = 1300

def pwn_spp_lon(locL, scaleL, num):
  possible_pwn_spp_lon1 = np.random.normal(locL, scaleL, size=mock_num)
  possible_pwn_spp_GLON_where = np.where(possible_pwn_spp_lon1 > 0., possible_pwn_spp_lon1, 
                                         possible_pwn_spp_lon1+360.)
  possible_selected_GLON_pwn_spp = possible_pwn_spp_GLON_where[(possible_pwn_spp_GLON_where>0.) & (possible_pwn_spp_GLON_where<360.)]
  return possible_selected_GLON_pwn_spp

possible_final_mock_pwn_spp_LON = pwn_spp_lon(paramsLON_pwn[1], paramsLON_pwn[2], mock_num)


### E100 PWN/SPP


E100hist_pwn_spp, E100bins_pwn_spp,_ = plt.hist(np.log10(v22LP_df_pwn_spp_snr['En_flux_100']), 
                                                bins=AGNs_spp_pwn_snr_bins, density=False, alpha=0.8, 
                                                edgecolor='navy', color='olive')


plt.clf()


E100hist_list_pwn_spp = E100hist_pwn_spp.tolist()
E100bins_pwn_spp_list = E100bins_pwn_spp.tolist()

# print (np.argmax(E100hist_pwn_spp), np.max(E100hist_pwn_spp))
# print ('minimum bin boundary: ', E100bins_pwn_spp[0], E100bins_pwn_spp[1])



selected_hist1_pwn = E100hist_list_pwn_spp[0:6]
 
selected_bins1_pwn = E100bins_pwn_spp_list[0:6]

first_threshold = 3.4e-13 
log_first_threshold = math.log10(first_threshold)

fermi_threshold = 2e-12
log_fermi_threshold = math.log10(fermi_threshold)

# ax1.axvline(x=math.log10(fermi_threshold), label='Fermi Threshold', linestyle='--')
# ax1.axvline(x=math.log10(first_threshold), label='F1 Threshold', color='red', )
# ax1.axhline(y=selected_hist1_pwn[1], xmin=0.05, xmax=0.35, color='grey')
# ax1.set_yscale('log')
# ax1.set_xlabel(r'$E_{100}\, \left[\mathrm{erg}\, \mathrm{cm}^{-2}\, \mathrm{s}^{-1}\right] $', fontsize=11)
# ax1.set_ylabel('Source Counts', fontsize=11)
# plt.legend(fontsize=10)



bin_size_pwn = selected_bins1_pwn[0]-selected_bins1_pwn[1] # same for all bins 
# print ('bin size: ', bin_size_pwn)
added_bins_pwn = []
for i in range(1,10):
  new_bin = selected_bins1_pwn[0] + (bin_size_pwn *i)
  # print (new_bin)
  if new_bin >= math.log10(first_threshold):
    added_bins_pwn.append(new_bin)
# print (added_bins_pwn, '\n', len(added_bins_pwn))



added_bins_pwn = added_bins_pwn[::-1]

added_bins_pwn.extend(E100bins_pwn_spp)

# print ('check new bins list: ', added_bins_pwn, len(added_bins_pwn))


# print (E100hist_pwn_spp[1], selected_hist1_pwn[1])
E100hist_pwn_high_L = E100hist_pwn_spp[1:]
# print (E100hist_pwn_high_L)

E100hist_list_pwn_added = [selected_hist1_pwn[1]]*3 
# # # multiplication factor = added bins (here it is 2) + from which entry highest counts were considered, here it is 1 
E100hist_list_pwn_added.extend(E100hist_pwn_high_L)

# print(E100hist_list_pwn_added, '\n', len(E100hist_list_pwn_added), '\n', len(added_bins_pwn))
# print (E100hist_list_pwn_spp)

num_cats = 1
start_source_num = 0

print ('reached before for loop num cats')

for i in range(num_cats):

    # np.random.seed(35) # change seed for different run
    noise_N_pwn_spp = np.random.uniform(0.7, 1.3, AGNs_spp_pwn_snr_bins)
    noise_N_pwn_spp_combined = np.random.uniform(0.8, 1.2, AGNs_spp_pwn_snr_bins+2)

    choose_number_pwn_spp = int(len(possible_final_mock_pwn_spp_LON) * random.choice(np.random.uniform(0.82, 0.92, num_cats)))
    print ('choose_number_pwn_spp: ', choose_number_pwn_spp)	

    #### Generation of pwn_spp Catalogue

    rand_LPIndex_mock_dist_pwn_spp    = random.sample(LPIndex_mock_dist_pwn_spp_snr.tolist(), choose_number_pwn_spp)
    rand_LPBeta_mock_dist_pwn_spp     = random.sample(LPBeta_mock_dist2_pwn_spp_snr.tolist(), choose_number_pwn_spp)
    rand_LPFDensity_mock_dist_pwn_spp = random.sample(LPFDensity_mock_dist_pwn_spp_snr.tolist(), choose_number_pwn_spp)
    rand_LPPivEn_mock_dist_pwn_spp    = random.sample(LPPivEn_mock_dist3_pwn_spp_snr.tolist(), choose_number_pwn_spp)

    rand_GLAT_pwn_spp                 = random.sample(total_histLAT1f_pwn_spp.tolist(), choose_number_pwn_spp)
    rand_GLON_pwn_spp                 = random.sample(possible_final_mock_pwn_spp_LON.tolist(), choose_number_pwn_spp)

    def simple_lum_pwn_spp_comb(n, LPalpha, LPbeta, LPFd, LPPEn, LPglat, LPglon):
        result_list_pwn_spp = []
        final_result_list_pwn_spp = []
        c_lp_a_pwn_spp   = [] # AGN alpha
        c_lp_b_pwn_spp   = [] # AGN beta
        c_lp_F_pwn_spp   = [] # flux density
        c_lp_PEn_pwn_spp = [] # pivot  energy
        c_GLAT_pwn_spp   = [] # lat
        c_GLON_pwn_spp   = [] # long
        #### loop over the number of sources (selected in random)
        for x in range(choose_number_pwn_spp):
            LP_index_pwn_spp = LPalpha[x]
            LP_beta_pwn_spp  = LPbeta[x]
            LP_FD_pwn_spp    = LPFd[x]
            LP_PEn_pwn_spp   = LPPEn[x]
            LP_glat_pwn_spp  = LPglat[x]
            LP_glon_pwn_spp  = LPglon[x]

            
            
            result_pwn_spp   = Integrate(n, 100, 100e3, alpha=LP_index_pwn_spp, beta=LP_beta_pwn_spp, 
                                    PivEn=LP_PEn_pwn_spp, FluxDensity=LP_FD_pwn_spp)
            # n is the number of steps in integration, higher ---> better accuracy
            # unit here is MeV cm^-2 s^-1 :
            result_pwn_spp   = result_pwn_spp * 1.6e-6 
            # unit here is Erg cm^-2 s^-1:
            # lowest bin -12.378695, highest bin -9.002079  
            # E100hist_list_bll_combined = check_area1sigh_bll_list_pow + E100hist_list_bll_high
            # E100bins_bll_list_combined = selected_bins1_list_bll_extended + E100bins_bll_list_high
            if result_pwn_spp>=10**added_bins_pwn[0] and result_pwn_spp < 10**(added_bins_pwn[-1]): # 10**(lbin), 10**(hbin)
                result_list_pwn_spp.append(result_pwn_spp)

                c_lp_a_pwn_spp.append(LP_index_pwn_spp)
                c_lp_b_pwn_spp.append(LP_beta_pwn_spp)
                c_lp_F_pwn_spp.append(LP_FD_pwn_spp)
                c_lp_PEn_pwn_spp.append(LP_PEn_pwn_spp)
                c_GLAT_pwn_spp.append(LP_glat_pwn_spp)
                c_GLON_pwn_spp.append(LP_glon_pwn_spp)
                
                mockE100hist_pwn_spp, bins_pwn_spp = np.histogram(np.log10(result_list_pwn_spp), bins=added_bins_pwn, density=False)
                mockE100hist_list_pwn_spp = mockE100hist_pwn_spp.tolist()
                for x1, x2, x3 in zip(mockE100hist_list_pwn_spp, E100hist_list_pwn_added, noise_N_pwn_spp_combined): 
                    if (x1*x3) > x2:
                        # print ('!!! mock higher than the real !!!', x1, x2)
                        result_list_pwn_spp.pop()
    #                     print ('result_list len: ', len(result_list))
                        final_result_list_pwn_spp = result_list_pwn_spp[:]
                        c_lp_a_pwn_spp.pop()
                        c_lp_b_pwn_spp.pop()
                        c_lp_F_pwn_spp.pop()
                        c_lp_PEn_pwn_spp.pop()
                        c_GLAT_pwn_spp.pop()
                        c_GLON_pwn_spp.pop()
            
                    else:
                        continue
                        print (':) :) what is happening here :) :)')
                    
    #                 result_list.pop()    
        return final_result_list_pwn_spp, c_lp_a_pwn_spp, c_lp_b_pwn_spp, c_lp_F_pwn_spp, c_lp_PEn_pwn_spp, c_GLAT_pwn_spp, c_GLON_pwn_spp

    final_result_list_pwn_spp_comb, c_lp_a_pwn_spp_comb, c_lp_b_pwn_spp_comb, c_lp_F_pwn_spp_comb, c_lp_PEn_pwn_spp_comb, c_GLAT_pwn_spp_comb, c_GLON_pwn_spp_comb = simple_lum_pwn_spp_comb(1070, 
                                                                                                                                                                                            rand_LPIndex_mock_dist_pwn_spp, 
                                                                                                                                                                                            rand_LPBeta_mock_dist_pwn_spp, 
                                                                                                                                                                                            rand_LPFDensity_mock_dist_pwn_spp, 
                                                                                                                                                                                            rand_LPPivEn_mock_dist_pwn_spp, 
                                                                                                                                                                                            rand_GLAT_pwn_spp, rand_GLON_pwn_spp)







    pwn_spp_dec_cord_list = []
    pwn_spp_ra_cord_list = []

    mock_source_num_pwn_spp_comb = [i for i in range(len(c_lp_a_pwn_spp_comb))]



    for plo, pla in zip(c_GLON_pwn_spp_comb, c_GLAT_pwn_spp_comb):
        c_icrs1_pl  = SkyCoord(l=plo*u.degree, b=pla*u.degree, frame='galactic')
        pwn_spp_ra_cord_list.append(c_icrs1_pl.fk5.ra.degree)
        pwn_spp_dec_cord_list.append(c_icrs1_pl.fk5.dec.degree)
    
    if (len(c_GLAT_pwn_spp_comb)<=215) and (len(c_GLAT_pwn_spp_comb)>=150):

        mock_source_num_pwn_spp_comb = [i for i in range(len(c_lp_a_pwn_spp_comb))]
        # print (len(mock_source_num_pwn_spp_comb))

        source_numbers_check = len(c_GLAT_pwn_spp_comb)
        print ('sources in file: ', source_numbers_check)

        mock_xmlfile_pwn_spp_comb = open('/content/drive/My Drive/Colab Notebooks/mock_4FGL_pwnF1_%d.xml' %(start_source_num), 'w')
        mock_xmlfile_pwn_spp_comb.write('<source_library title="source library">\n')
        for n, al, be, pivE, ra, dec, num in zip(c_lp_F_pwn_spp_comb, c_lp_a_pwn_spp_comb, c_lp_b_pwn_spp_comb, 
                                                c_lp_PEn_pwn_spp_comb, 
                                                pwn_spp_ra_cord_list, pwn_spp_dec_cord_list, 
                                                mock_source_num_pwn_spp_comb):
            
            mock_xmlfile_pwn_spp_comb.write('<source name="LogParabola_source{0}" type="PointSource">\n'.format(num))
            mock_xmlfile_pwn_spp_comb.write('<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n')
            mock_xmlfile_pwn_spp_comb.write('<spectrum type="LogParabola">\n')
            mock_xmlfile_pwn_spp_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="norm" scale="1.0" value="{2}"/>\n'.format(max(c_lp_F_pwn_spp_comb), min(c_lp_F_pwn_spp_comb), n))
            mock_xmlfile_pwn_spp_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="alpha" scale="1.0" value="{2}"/>\n'.format(max(c_lp_a_pwn_spp_comb), min(c_lp_a_pwn_spp_comb), al))
            mock_xmlfile_pwn_spp_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="Eb" scale="1" value="{2}"/>\n'.format(max(c_lp_PEn_pwn_spp_comb), min(c_lp_PEn_pwn_spp_comb), pivE))
            mock_xmlfile_pwn_spp_comb.write(
                '<parameter free="1" max="{0}" min="{1}" name="beta" scale="1.0" value="{2}"/>\n'.format(max(c_lp_b_pwn_spp_comb), min(c_lp_b_pwn_spp_comb), be))
            mock_xmlfile_pwn_spp_comb.write('</spectrum>\n')
            mock_xmlfile_pwn_spp_comb.write('<spatialModel type="SkyDirFunction">\n')
            mock_xmlfile_pwn_spp_comb.write(
                '<parameter free="0" max="360." min="0." name="RA" scale="1.0" value="{0}"/>\n'.format(ra))
            mock_xmlfile_pwn_spp_comb.write(
                '<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{0}"/>\n'.format(dec))
            mock_xmlfile_pwn_spp_comb.write('</spatialModel>\n')
            mock_xmlfile_pwn_spp_comb.write('</source>\n')
            
        mock_xmlfile_pwn_spp_comb.write('</source_library>')    
        mock_xmlfile_pwn_spp_comb.close()

    else: 
        continue
        print ('!!! continue but it should not reach here')
    start_source_num = start_source_num + 1
