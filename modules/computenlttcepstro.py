import numpy as np
from scipy import signal
import pickle as pk
import os
try:
    import sportran as st
except ImportError:
    from sys import path
    path.append('..')
    import sportran as st
from sportran import md
import time


def autocorr(x):
    result = signal.correlate(x, x, mode='full', method='fft')
    v = [result[i] / (len(x) - abs(i - (len(x)) + 1)) for i in range(len(result))]
    return np.array(v[int(result.size / 2):])

def Ggeneratemod(nk):
    G = np.zeros((nk, 3))
    conta = 0
    i1 = 1
    i2 = 0
    i3 = 0
    G[0] = np.array([0, 0, 0])
    for i in range(1, nk):
        G[i] = np.array([i1, i2, i3])
        if G[i][0] != G[i][1] and G[i][1] == G[i][2]:
            i2 += 1
            if G[i][0] != G[i][1] and G[i][1] != G[i][2]:
                i3 += 1
        else:
            if G[i][1] != G[i][2]:
                i3 += 1
        if G[i][0] == G[i][1] and G[i][0] == G[i][2]:
            i1 += 1
            i2 = 0
            i3 = 0
    Gmod = np.linalg.norm(G, axis=1)
    return Gmod


def Ggeneratemodall(nk, L):
    G = np.zeros((nk, 3))
    conta = 0
    G[0] = np.array([0, 0, 0]) + 2.335581758729501e-06 / 2 / np.pi / np.sqrt(3.0)
    nkp = int(np.power(nk, 1 / 3)) + 1
    for i in range(0, nkp):
        for j in range(0, nkp):
            for k in range(0, nkp):
                if i==0 and j==0 and k==0 : continue
                conta += 1
                if conta == nk:
                    Gmod = np.linalg.norm(G, axis=1)
                    return Gmod
                G[conta] = np.array([i, j, k]) / L + 2.335581758729501e-06 / 2 / np.pi / np.sqrt(3.0)

    Gmod = np.linalg.norm(G, axis=1)
    return Gmod

def computenlttcepstro(root, Np, L, nk, nkk, cp, deltat, tdump, nskip=1):

    #with open(root + 'enk.pkl', 'rb') as f:
    #    enk = pk.load(f)
    #print(root + 'enk.pkl' + 'loaded correctly')
    #enka = np.array(enk)
    #enk = 0
    enka=np.load(root + 'enk.npy')
    print('list transformed in np.array')
    kpoints = []
    wcepstrum = []
    swcepstrum = []
    print('start loop up to nk')
    rho = Np / (6.022e23 * L[0] * L[1] * L[2] * 1.e-30)  # mol/m^3

    fac = rho * cp  # J/k/m^3

    dt = deltat * tdump  # ps
    Gmod = Ggeneratemodall(nk, L)
    for k in range(1, nkk, nskip):
        start = time.time()
        f = open(root + 'enk{}.dat'.format(k), 'w')
        f.write('c_enk[1] c_enk[2] \n')
        for i in enka[:, k]:
            f.write('{} {}\n'.format(np.real(i), np.imag(i)))
        f.close()



        jfile = st.i_o.TableFile(root + 'enk{}.dat'.format(k), group_vectors=True)

        jfile.read_datalines(start_step=0, NSTEPS=0, select_ckeys=['enk'])

        DT_FS = 1

        j = st.Current([jfile.data['enk']], DT_FS=DT_FS,
                       KAPPA_SCALE=1)
        fstar_THz = j.Nyquist_f_THz / 4
        jf = j.resample(fstar_THz=fstar_THz)
        print(j.Nyquist_f_THz)
        j.cepstral_analysis()
        jf.cepstral_analysis()
        j.compute_acf()
        kpoints.append(2 * Gmod[k] * np.pi)
        wcepstrum.append(j.acfm[0] / (jf.dct.tau_Kmin * (2 * Gmod[k] * np.pi) ** 2) * fac / dt * (
            1e-10) ** 2 / 1e-12)
        swcepstrum.append(j.dct.tau_std_Kmin / jf.dct.tau_Kmin * j.acfm[0] / (
                    jf.dct.tau_Kmin * (2 * Gmod[k] * np.pi) ** 2) * fac / dt * (1e-10) ** 2 / 1e-12)
        print(k,2 * Gmod[k] * np.pi,
              j.acfm[0] / (jf.dct.tau_Kmin * (2 * Gmod[k] * np.pi) ** 2) * fac / dt * (
                  1e-10) ** 2 / 1e-12,
              j.dct.tau_std_Kmin / jf.dct.tau_Kmin * j.acfm[0] / (
                      jf.dct.tau_Kmin * (2 * Gmod[k] * np.pi) ** 2) * fac / dt * (1e-10) ** 2 / 1e-12
              )
        os.remove(root + 'enk{}.dat'.format(k))
        print('0', start - time.time())
    return np.array(kpoints), np.array(wcepstrum), np.array(swcepstrum)



