import matplotlib.pyplot as plt
from modules import initialize
from modules import tools
from modules import molar
import numpy as np
import pickle as pk
import os


def seebeck(filename='dump.lammpstrj', root='./', posox='0.', nk=100, ntry=-1, filename_loglammps='log.lammps',
            plot=False, UNITS='metal', nblocks=12):
    inp = initialize.getinitialize(filename=filename,
                                   root=root,
                                   posox=posox,
                                   nk=nk,
                                   ntry=ntry)

    G = tools.Ggeneratemodall(inp['number of k'], inp['size']) * 2 * np.pi

    if not os.path.exists(root + filename_loglammps + '.npy'):
        log = molar.read_log_lammps(root=inp['root'],
                                    filename=filename_loglammps)
    else:
        log = np.load(root + filename_loglammps + '.npy', allow_pickle='TRUE').item()


    print(log['Temp'].mean(), 'K ', log['Temp'].std() / np.sqrt(len(log['Temp'])), 'K ',
          log['Press'].mean(), 'atm', log['Press'].std() / np.sqrt(len(log['Press'])), 'atm')

    hp = molar.molar_enthalpy(root=inp['root'],
                              filename=inp['filename'],
                              filename_log=filename_loglammps,
                              volume=inp['size'].prod(),
                              Np=inp['N'],
                              nblocks=nblocks,
                              UNITS=UNITS)

    h = hp.mean(axis=0).mean(axis=0)

    with open(inp['root'] + 'enk.pkl', 'rb') as g:
        enkb = pk.load(g)
        enk = np.array(enkb)
        enkb = 0
    with open(inp['root'] + 'chk.pkl', 'rb') as g:
        chkb = pk.load(g)
        chk = np.array(chkb)
        chkb = 0
    with open(inp['root'] + 'n1k.pkl', 'rb') as g:
        n1kb = pk.load(g)
        n1k = np.array(n1kb)
        n1kb = 0
    with open(inp['root'] + 'n2k.pkl', 'rb') as g:
        n2kb = pk.load(g)
        n2k = np.array(n2kb)
        n2kb = 0

    a = np.zeros(chk.shape[1] - 1, np.complex_)
    b = np.zeros(chk.shape[1] - 1, np.complex_)
    c = np.zeros(chk.shape[1] - 1, np.complex_)
    va = np.zeros(chk.shape[1] - 1)
    vb = np.zeros(chk.shape[1] - 1)
    vc = np.zeros(chk.shape[1] - 1)

    if UNITS == 'metal':
        fac = (16.022 * 1.0e-30 * 1.60218e-19 * 1.0e-10 /
               (inp['size'].prod() * 1.0e-30 * 1.38e-23 * log['Temp'].mean() ** 2 * 8.854 * 1.0e-12))/4/np.pi
    if UNITS == 'real':
        fac = (16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 /
               (inp['size'].prod() * 1.0e-30 * 1.38e-23 * log['Temp'].mean() ** 2 * 8.854 * 1.0e-12))/4/np.pi
    if UNITS == 'lj':
        fac = (1/(inp['size'].prod() * log['Temp'].mean() ** 2))

    face = (16.022 * 1.0e-20 * 1.0e-10) * (16.022 * 1.0e-20 * 1.0e-10)/ (inp['size'].prod() * 1.0e-30 * 1.38e-23 * log['Temp'].mean() * 8.854 * 1.0e-12)
    print('static dielectric constant')
    print((np.mean((chk) * np.conj(chk), axis=0) / G ** 2)[0] * face)
    print('polarizability at k_min')
    print((np.mean((chk) * np.conj(chk), axis=0) / G ** 2)[1] * face)
    print('enthalpy contribution')
    print(-(np.mean((n1k * h[0] + n2k * h[1]) * np.conj(chk), axis=0) / G ** 2)[1] * fac)

    #np.savetxt(root+'convergenceenergy_charge.out', np.real((((enk) * np.conj(chk))/ G ** 2)[:, 1]) * fac)
    #np.savetxt(root + 'convergenceseebeck.out', np.real((((enk - n1k * h[0] - n2k * h[1]) * np.conj(chk)) / G ** 2)[:, 1]) * fac)
    a = (np.mean((enk) * np.conj(chk), axis=0) / G ** 2)[1:] * fac
    b = -(np.mean((n1k * h[0] + n2k * h[1]) * np.conj(chk), axis=0) / G ** 2)[1:] * fac
    for i in range(1, chk.shape[1]):
        std, bins = tools.stdblock(fac * (enk[:, i]) * np.conj(chk[:, i]) / G[i] ** 2)
        pp = int(16 * len(std) / 20)
        va[i - 1] = std[pp]
        std, bins = tools.stdblock(
            fac * (n1k[:, i] * h[0] + h[1] * n2k[:, i]) * np.conj(chk[:, i]) / G[i] ** 2)
        pp = int(16 * len(std) / 20)
        vb[i - 1] = std[pp]
    with open(inp['root'] + 'stdk_min.out', 'w') as f:

        stda, bins = tools.stdblock(fac * (enk[:, 1]) * np.conj(chk[:, 1]) / G[1] ** 2)
        stdb, bins = tools.stdblock(
            fac * (n1k[:, 1] * h[0] + h[1] * n2k[:, 1]) * np.conj(chk[:, 1]) / G[1] ** 2)
        for i in range(len(bins)):
            f.write('{}\t'.format(bins[i]) + '{}\n'.format((stda + stdb)[i]))

    if plot:
        f, ax = plt.subplots(1)
        ax.errorbar(G[1:] * 10, a + b, np.sqrt(va + vb), fmt='.')

        f, ax = plt.subplots(3, figsize=[8, 12])

        ax[0].errorbar(G[1:] * 10, b, np.sqrt(vb), fmt='.', label='contributo delle entalpie')
        ax[0].legend()
        ax[0].set_xlabel(r'k [$nm^{-1}$]')
        ax[0].set_ylabel(r'S_enth [$\frac{V}{K}$]')
        ax[1].errorbar(G[1:] * 10, a, np.sqrt(va), fmt='.', label='contributo delle energie')
        ax[1].legend()
        ax[1].set_xlabel(r'k [$nm^{-1}$]')
        ax[1].set_ylabel(r'S_en [$\frac{V}{K}$]')
        ax[2].errorbar(G[1:] * 10, a + b, np.sqrt(va + vb), fmt='.', label='totale')
        ax[2].legend()
        ax[2].set_xlabel(r'k [$nm^{-1}$]')
        ax[2].set_ylabel(r'S [$\frac{V}{K}$]')


    with open(inp['root'] + 'seebeck.out', 'w') as f:
        for i in range(chk.shape[1] - 1):
            f.write(
                '{}\t'.format(G[i + 1] * 10) + '{}\t'.format(np.real(a + b)[i]) + '{}\n'.format(np.sqrt(va + vb)[i]))
    return

def kcorr( ak, bk, filename='dump.lammpstrj', root='./', posox='0.', nk=100, ntry=-1, filename_loglammps='log.lammps',
            plot=False, UNITS='metal', nblocks=12):
    inp = initialize.getinitialize(filename=filename,
                                   root=root,
                                   posox=posox,
                                   nk=nk,
                                   ntry=ntry)

    G = tools.Ggeneratemodall(inp['number of k'], inp['size']) * 2 * np.pi


    a = np.zeros(ak.shape[1] - 1, np.complex_)
    va = np.zeros(ak.shape[1] - 1)


    a = (np.mean((ak) * np.conj(bk), axis=0) / G ** 2)[1:]

    stdc = tools.stdblock_parallel(((ak[:, 1:]) * np.conj(bk[:, 1:]) / G[1:] ** 2).T, ncpus=40)
    std = stdc[:, 0, int(stdc.shape[2]/2)]

    return a, std