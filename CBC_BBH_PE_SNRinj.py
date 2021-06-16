from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import bilby
import logging
import corner
import numpy as np
import pandas as pd
import h5py
import json
import math
from astropy.cosmology import FlatLambdaCDM

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager
import platform

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

# ## Definign the dataframe to write the data of Uniformly distibuted BBH.
# ## Use this part of code after finishing the run to Check if SNR > 10 or SNR < 10.
# ## Change the directory where you have the data
# column = ['Injection_idx', 'ET-D_1', 'ET-D_2', 'ET-D_3', 'Redshift', 'Luminosity Distance (Mpc)']
# Injection_idx= open('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/Uniform_BBH_ET-Only_SNRg10.txt','r')
# # CE = open('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/CE_SNR.txt','r')
# ET1 = open('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/ET1_SNR.txt','r')
# ET2 = open('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/ET2_SNR.txt','r')
# ET3 = open('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/ET3_SNR.txt','r')
# Redshift = open('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/Redshift_SNR.txt','r')
# Luminosity_dist = open('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/Luminosity_dist.txt','r')
#
# Injection_idx = Injection_idx.readline().split()
# # CE = CE.read().split()
# ET1 = ET1.read().split()
# ET2 = ET2.read().split()
# ET3 =ET3.read().split()
# Redshift = Redshift.read().split()
# Luminosity_dist =Luminosity_dist.read().split()
#
# data_inj = {'Injection_idx' : Injection_idx, 'ET-D_1': ET1, 'ET-D_2':ET2, 'ET-D_3':ET3,
#             'Redshift':Redshift, 'Luminosity Distance (Mpc)':Luminosity_dist}
# data_SNR = pd.DataFrame(data_inj, columns=column)
# data_SNR.to_csv('./Uniform_BBH_DATA/BBH_ET-Only_SNRg10/Uniform_BBH_ET_Only_SNRg10_datasets.csv')

# file = open('Waveform1_IMRPhenomv2_Uni_BBH_SNRg10.txt', 'r')
# file = file.readline().split()

# save_idx = open(r'Waveform_SEOBNRv4P_Uni_BBH_SNRg10.txt', 'w')

CE_SNR = open(r'CE_SNR.txt', 'w')
ET1_SNR = open(r'ET1_SNR.txt', 'w')
ET2_SNR = open(r'ET2_SNR.txt', 'w')
ET3_SNR = open(r'ET3_SNR.txt', 'w')
redshift = open(r'Redshift_SNR.txt','w')
lum_dist = open(r'Luminosity_dist.txt','w')

injectionPE = False

## IMRPhenomPv2 and SEOBNRv4P BBH SNR > 10.
IDXs = [0, 4, 12, 30, 40, 41, 42, 53, 57, 64,
        70, 71, 73, 74, 87, 92, 94, 109, 125, 127,
        130, 132, 142, 143, 148, 150, 154, 159, 167, 171,
        177, 183, 184, 187, 196, 202, 206, 208, 209, 214,
        217, 218, 226, 231, 235, 244, 258, 264, 275, 279,
        282, 283, 285, 288, 291, 292, 298, 301, 302, 303,
        307, 309, 312, 315, 322, 324, 332, 334, 336, 337,
        340, 342, 345, 346, 347, 352, 357, 362, 369, 370,
        381, 383, 384, 389, 392, 393, 396, 397, 401, 402,
        404, 405, 406, 407, 408, 409, 411, 419, 427, 430,
        442, 448, 450, 454, 456, 458, 463, 480, 488, 492,
        500, 501, 502, 505, 509, 510, 517, 522, 523, 525,
        528, 534, 541, 543, 544, 548, 556, 557, 560, 562,
        566, 568, 576, 582, 587, 588, 590, 592, 594, 596,
        600, 601, 604, 607, 608, 610, 617, 625, 627, 629  ]

for idx in IDXs:
# for idx in file:
# for idx in np.arange(1000):

    # idx = int(idx)
    print('idx is', idx)

    ifos = ['CE', 'ET_D_TR']
    sampler = 'dynesty'

    # Specify the output storage directory.
    outdir = 'outdir'
    label = 'sample_param_' + str(idx)
    extension='json'
    bilby.utils.setup_logger(outdir=outdir, label=label)

    ## Duration and sampling frequency.
    duration = 4.                     # units = sec.
    sampling_frequency = 2048.        # units =  Hz.

    ## Loading the injection parameters from hdf5 data file.
    injection_parameters = pd.read_hdf('./Injection_file/injections_10e6.hdf5')
    injection_parameters = dict((injection_parameters).loc[idx])

    # injection_parameters = injection_parameters.iloc[IDXs]
    # injection_parameters.index = range(len(injection_parameters.index))

    ## Changing 'iota' to 'theta_jn' to be suitable with bilby
    if 'iota' in injection_parameters:
        injection_parameters['theta_jn'] = injection_parameters.pop('iota')

    ## we use start_time=start_time to match the start time and wave interferome time. if we do not this then it will create mismatch between time.
    start_time = injection_parameters['geocent_time']
    ## Redshift to luminosity Distance conversion using bilby
    injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(injection_parameters['redshift'])


    # ## Luminosity Distance
    # font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
    # font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')
    #
    # sns.distplot(injection_parameters['luminosity_distance'] / 1000, bins=20, hist=True, kde=True,label='True $d_{L}~$[Gpc]')
    # legend = plt.legend(loc='best', prop=font1)
    # plt.xlabel(' Luminosity Distance $d_{L}~[Gpc]$', fontsize=14)
    # plt.ylabel('Probability Distribution P($d_{L}$)', fontsize=14)
    # plt.tight_layout()
    # plt.savefig('Luminosity_Dist_BBH_SNRg10', dpi=300)
    # plt.close()


    # First mass needs to be larger than second mass
    if injection_parameters['mass_1'] < injection_parameters['mass_2']:
        tmp = injection_parameters['mass_1']
        injection_parameters['mass_1'] = injection_parameters['mass_2']
        injection_parameters['mass_2'] = tmp

    ## Fixed arguments passed into the source model : A dictionary of fixed keyword arguments
    ## to pass to either `frequency_domain_source_model` or `time_domain_source_model`.
    waveform_arguments = dict(waveform_approximant= 'IMRPhenomPv2', #'SEOBNRv4P',
                              reference_frequency=50., minimum_frequency=2.)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency, start_time=start_time,
          frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, waveform_arguments=waveform_arguments)

    ## Initialization of GW interferometer
    IFOs = bilby.gw.detector.InterferometerList(ifos)

    # Generates an Interferometer with a power spectral density based on advanced LIGO.
    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)
    IFOs.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    mf_snr = np.zeros((1, len(IFOs)))[0]
    waveform_polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
    k = 0
    for ifo in IFOs:
        signal_ifo = ifo.get_detector_response(waveform_polarizations, injection_parameters)
        ## Calculate the _complex_ matched filter snr of a signal.
        ##This is <signal|frequency_domain_strain> / optimal_snr
        mf_snr[k] = ifo.matched_filter_snr(signal=signal_ifo)
        if np.isnan(mf_snr[k]):
            mf_snr[k] = 0.

        print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, mf_snr[k], injection_parameters['redshift']))

        if k == 0:
            CE_SNR.write('{} '.format(mf_snr[k]) + '\n')
            redshift.write('{}'.format(str(injection_parameters['redshift'])) + '\n')
            lum_dist.write('{}'.format(str(injection_parameters['luminosity_distance'])) + '\n')
        if k == 1:
            ET1_SNR.write('{}'.format(mf_snr[k]) + '\n')
        if k == 2:
            ET2_SNR.write('{}'.format(mf_snr[k]) + '\n')
        if k == 3:
            ET3_SNR.write('{}'.format(mf_snr[k]) + '\n')

        k += 1

    if np.all(mf_snr > 10):

        injectionPE =True

        # save_idx.write('{} '.format(idx))
        print(' Hoorahhh! SNR is greater than 10 for idx', idx)
        # print(' Hoorahhh! SNR is less than 10 for idx', idx)

        # # Setup prior
        # priors = bilby.gw.prior.BBHPriorDict(filename='binary_black_holes_cosmo_uniform.prior')
        # pzDlBH = np.loadtxt("PzDlBH.txt", delimiter=',')
        #
        # priors['geocent_time'] = bilby.core.prior.Uniform( minimum=injection_parameters['geocent_time'] - 1,
        #            maximum=injection_parameters['geocent_time'] + duration, name='geocent_time', latex_label='$t_c$', unit='$s$')
        #
        # priors['luminosity_distance'] = bilby.core.prior.Interped(
        #      name='luminosity_distance', xx=pzDlBH[:, 1], yy=pzDlBH[:, 2], minimum=1e1, maximum=1e5, unit='Mpc')
        #
        # #priors['luminosity_distance'] = bilby.core.prior.Uniform(
        #  #       name='luminosity_distance', minimum=1e2, maximum=1e4, unit='Mpc')  # Here change the max value of Luminosity Distnace
        #
        # ## For parameters vary only in mass_1 and mass_2 and rest of all are fixed : use this one
        # # priors['mass_1'] = bilby.core.prior.PowerLaw(
        # #        name='mass_1', alpha=-1, minimum=5, maximum=50, unit='$M_{\\odot}$')  ## One can also use bilby.core.prior.Uniform depending on their own interest.
        # # priors['mass_2'] = bilby.core.prior.PowerLaw(
        # #        name='mass_2', alpha=-1, minimum=5, maximum=50, unit='$M_{\\odot}$')
        # # priors["dec"] = bilby.core.prior.Cosine(name='dec')
        # # priors["ra"] = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi)
        # # priors["theta_jn"] = bilby.core.prior.Sine(name='theta_jn')
        # # priors["psi"] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi)
        # # priors["phase"] = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi)
        #
        # ## list of injection_parameters key that are not being sampled
        # for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'geocent_time']:
        #     priors[key] = injection_parameters[key]
        #
        # del priors['redshift']
        # print('priors',priors)
        # logging.info(priors.keys())
        #
        # # Initialise the likelihood by passing in the interferometer data (IFOs) and the waveform generator
        # likelihood = bilby.gw.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator,
        #                     time_marginalization=False, phase_marginalization=False, distance_marginalization=False, priors=priors)
        #
        # sampling_seed = np.random.randint(1, 1e6)
        # np.random.seed(sampling_seed)
        # logging.info('Sampling seed is {}'.format(sampling_seed))
        #
        # sampler_dict = dict()
        # sampler_dict['dynesty'] = dict(npoints=1000)
        # sampler_dict['pymultinest'] = dict(npoints=1000, resume=False)
        # sampler_dict['emcee'] = {'nwalkers': 40, 'nsteps': 20000, 'nburn': 2000}
        #
        # result = bilby.core.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=100,
        #                               injection_parameters=injection_parameters, outdir=outdir, label=label,
        #                               result_class=bilby.gw.result.CBCResult)
        #
        # ## make some plots of the outputs
        # result.plot_corner()
        # plt.close()
        # result.plot_skymap(maxpts=500)
        # plt.close()

    else:
        print('SNR is less then 10 for idx : ', idx)
        # print('SNR is greater then 10 for idx : ', idx)
