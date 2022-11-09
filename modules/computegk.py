import sportran as st
import matplotlib.pyplot as plt
import numpy as np
from modules import initialize
from modules import tools
from modules import computenlttcepstro
from modules import seebeck_work
from modules import molar
from multiprocessing import Pool
import time
import os

def computecorrflux(root, filename, outname, nk, flux1, flux2, nblock=40, ncpus=40):
    inp = initialize.getinitialize(filename, root, 0, nk, -1)
    print('DEFAULT: METAL UNITS')

    c = np.real(tools.corr_parallel(np.mean(flux1.T, axis=0),
                                np.mean(flux2.T, axis=0),
                                nblock, ncpus=ncpus))
    res = {}
    res = {'correlation function': c}
    np.save(root+outname+'.npy', res)
    return c

def computegkflux(root, filename, outname, nk, flux1, flux2, nblocks=[40], ncpus=40):


    c = [[] for i in nblocks]
    trc = [[] for i in nblocks]
    stdc = [[] for i in nblocks]
    hetrc = [[] for i in nblocks]
    hestdc = [[] for i in nblocks]
    t = [[] for i in nblocks]

    for i in range(len(nblocks)):
        nblock = nblocks[i]

        if os.path.exists(root+outname+'.npy'):
            res = np.load(root+outname+'.npy', allow_pickle=True).item()
            c[i] = res['correlation function']
            res = 0
        else:
            c[i] = computecorrflux(root, filename, outname, nk, flux1, flux2, nblock=nblock, ncpus=ncpus)

        tau = len(c[i][0, :])

        # HE
        hetrc[i] = np.zeros((nblock, int(tau)))
        start = time.time()
        gkc = np.cumsum(c[i][:, 1:], axis=1)
        t[i] = np.linspace(1, tau, tau-1)
        hc = np.cumsum(c[i][:, 1:] * t[i], axis=1)

        # HE : \int_0^\tau <j(t)j(0)>(1-t/\tau)
        for j in range(1, int(tau - 1)):
            hetrc[i][:, j - 1] = (gkc[:, j - 1] - hc[:, j - 1] / j + c[i][:, 0] / 2)


        hestdc[i] = tools.stdblock_parallel(hetrc[i].T, ncpus=ncpus)

    res = {}
    res = {'time': t,
           'cc_onsager': hetrc,
           'cc_std': hestdc}
    return res

def computecorrheat(root, filename, filename_loglammps, nk, redor=False, nblock=40, UNITS='metal'):
    inp = initialize.getinitialize(filename, root, 0, nk, -1)
    print('DEFAULT: METAL UNITS')
    if not os.path.exists(root + 'flux.npy') or redor:
        jfile = st.i_o.TableFile(root + 'gk' + filename_loglammps, group_vectors=True)

        jfile.read_datalines(start_step=0, NSTEPS=0, select_ckeys=['Temp', 'flux', 'Lx', 'vcm[1]', 'vcm[2]'])

        np.save(root + 'flux.npy', jfile.data['flux'])
        np.save(root + 'vcm[1].npy', jfile.data['vcm[1]'])
        np.save(root + 'vcm[2].npy', jfile.data['vcm[2]'])
        np.save(root + 'temp.npy', jfile.data['Temp'])
        flux = np.load(root + 'flux.npy') - np.load(root + 'flux.npy').mean()
        temp = np.load(root + 'temp.npy')
        vcm_1 = np.load(root + 'vcm[1].npy') * inp['N']/2 - np.load(root + 'vcm[1].npy').mean() * inp['N']/2
        vcm_2 = np.load(root + 'vcm[2].npy') * inp['N']/2 - np.load(root + 'vcm[2].npy').mean() * inp['N']/2
    else:
        flux = np.load(root + 'flux.npy') - np.load(root + 'flux.npy').mean()
        temp = np.load(root + 'temp.npy')
        vcm_1 = np.load(root + 'vcm[1].npy') * inp['N']/2 - np.load(root + 'vcm[1].npy').mean() * inp['N']/2
        vcm_2 = np.load(root + 'vcm[2].npy') * inp['N']/2 - np.load(root + 'vcm[2].npy').mean() * inp['N']/2
    TEMPERATURE = temp.mean()
    h = molar.molar_enthalpy(root, filename, filename_loglammps, inp['size'].prod(), inp['N'], nblocks=12, UNITS=UNITS)
    hm = h.mean(axis=1).mean(axis=0)

    ncpus = 40
    cec = np.real(tools.corr_parallel(np.mean(flux.T - hm[0] * vcm_1.T - hm[1] * vcm_2.T, axis=0),
                                np.mean(vcm_1.T - vcm_2.T, axis=0),
                                nblock, ncpus=ncpus))
    ccc = np.real(tools.corr_parallel(np.mean(vcm_1.T - vcm_2.T, axis=0),
                                np.mean(vcm_1.T - vcm_2.T, axis=0),
                                nblock, ncpus=ncpus))
    cee = np.real(tools.corr_parallel(np.mean(flux.T - hm[0] * vcm_1.T - hm[1] * vcm_2.T, axis=0),
                                np.mean(flux.T - hm[0] * vcm_1.T - hm[1] * vcm_2.T, axis=0),
                                nblock, ncpus=ncpus))
    res = {}
    res = {'heat charge currents time correlation': cec,
           'charge charge currents time correlation' : ccc,
           'heat heat currents time correlation' : cee}
    np.save(root+str(nblock)+'corr.npy', res)
    return

def computegk(root, filename, filename_loglammps, nk, redor=False, nblocks=[40], UNITS='metal', plot=False):
    ncpus = 40

    cec = [[] for i in nblocks]
    ccc = [[] for i in nblocks]
    trec = [[] for i in nblocks]
    trcc = [[] for i in nblocks]
    stdec = [[] for i in nblocks]
    stdcc = [[] for i in nblocks]
    tpgkc = [[] for i in nblocks]
    stdtpgkc = [[] for i in nblocks]
    hetrec = [[] for i in nblocks]
    hetrcc = [[] for i in nblocks]
    hestdec = [[] for i in nblocks]
    hestdcc = [[] for i in nblocks]
    hetpgkc = [[] for i in nblocks]
    hestdtpgkc = [[] for i in nblocks]
    t = [[] for i in nblocks]

    for i in range(len(nblocks)):
        nblock = nblocks[i]

        try:
            diccorr=np.load(root+str(nblock)+'corr.npy', allow_pickle=True).item()
        except:
            computecorrheat(root, filename, filename_loglammps, nk, nblock=nblock, redor=False)
            diccorr = np.load(root + str(nblock) + 'corr.npy', allow_pickle=True).item()
        temp = np.load(root + 'temp.npy')
        TEMPERATURE = temp.mean()
        cec[i] = diccorr['heat charge currents time correlation']
        ccc[i] = diccorr['charge charge currents time correlation']
        diccorr = 0

        tau = len(ccc[i][0, :])

        # GK
        trec[i] = -np.cumsum(cec[i][:, :], axis=1) / TEMPERATURE
        trcc[i] = np.cumsum(ccc[i][:, :], axis=1)

        stdcc[i] = tools.stdblock_parallel(trcc[i].T, ncpus=ncpus)
        stdec[i] = tools.stdblock_parallel(trec[i].T, ncpus=ncpus)

        tpgkc[i] = trec[i].mean(axis=0) / trcc[i].mean(axis=0)
        stdtpgkc[i] = np.sqrt(stdec[i][:, 0, 0] / trcc[i].mean(axis=0) ** 2 +
                              stdcc[i][:, 0, 0] * trec[i].mean(axis=0) ** 2 / trcc[i].mean(axis=0) ** 4)

        # HE
        hetrec[i] = np.zeros((nblock, int(tau)))
        hetrcc[i] = np.zeros((nblock, int(tau)))
        start = time.time()
        gkec = np.cumsum(cec[i][:, 1:], axis=1)
        gkcc = np.cumsum(ccc[i][:, 1:], axis=1)
        t[i] = np.linspace(1, tau, tau - 1)
        hec = np.cumsum(cec[i][:, 1:] * t[i], axis=1)
        hcc = np.cumsum(ccc[i][:, 1:] * t[i], axis=1)

    # HE : \int_0^\tau <j(t)j(0)>(1-t/\tau)
        for j in range(1, int(tau - 1)):
            hetrec[i][:, j - 1] = -(gkec[:, j - 1] - hec[:, j - 1] / j + cec[i][:, 0] / 2) / TEMPERATURE
            hetrcc[i][:, j - 1] = gkcc[:, j - 1] - hcc[:, j - 1] / j + ccc[i][:, 0] / 2

        print(time.time() - start)

        hestdcc[i] = tools.stdblock_parallel(hetrcc[i].T, ncpus=ncpus)
        hestdec[i] = tools.stdblock_parallel(hetrec[i].T, ncpus=ncpus)

        hetpgkc[i] = hetrec[i].mean(axis=0) / hetrcc[i].mean(axis=0)
        hestdtpgkc[i] = np.sqrt(hestdec[i][:, 0, 0] / hetrcc[i].mean(axis=0) ** 2 + \
                                hestdcc[i][:, 0, 0] * hetrec[i].mean(axis=0) ** 2 / hetrcc[i].mean(axis=0) ** 4)

    res={}
    res={'time': t,
        'cc_onsager': hetrcc,
        'cc_std': hestdcc ,
        'ec_onsager': hetrec,
        'ec_std': hestdec,
        'ec_cc_ratio_onsager': hetpgkc,
        'ec_cc_ratio_std': hestdtpgkc}

    return res