import numpy as np
from scipy import signal
import pickle as pk
import os
from modules import tools
from multiprocessing import Pool
try:
    import sportran as st
except ImportError:
    from sys import path
    path.append('..')
    import sportran as st
from sportran import md
import time






def computenlttcepstro(root, Np, L, nk, nkk, cp, deltat, tdump, nskip=1, enkainp=None, enkadata=None):
    if enkainp==None:
        try:
            enk = np.load(root + 'enka.npy').T
            chk = np.load(root + 'chka.npy').T
            n1k = np.load(root + 'n1ka.npy').T
            n2k = np.load(root + 'n2ka.npy').T
        except:
            with open(root + 'enk.pkl', 'rb') as g:
                enkb = pk.load(g)
                enk = np.array(enkb).T
                enkb = 0
            with open(root + 'chk.pkl', 'rb') as g:
                chkb = pk.load(g)
                chk = np.array(chkb).T
                chkb = 0
            with open(root + 'n1k.pkl', 'rb') as g:
                n1kb = pk.load(g)
                n1k = np.array(n1kb).T
                n1kb = 0
            with open(root + 'n2k.pkl', 'rb') as g:
                n2kb = pk.load(g)
                n2k = np.array(n2kb).T
                n2kb = 0
        enka = (enk - enk[0,:, np.newaxis]/Np*(n1k+n2k)).T
    else:
        enka = enkadata

    kpoints = []
    wcepstrum = []
    swcepstrum = []
    print('start loop up to nkk')
    with open(root + 'nlttkcepstral.out', 'w+') as f:
        pass
    for k in range(1, nkk, nskip):
        kpoint, tr, ertr = computenlttcepstro_k(root=root,
                                                Np=Np,
                                                L=L,
                                                nk=nk,
                                                cp=cp,
                                                deltat=deltat,
                                                tdump=tdump,
                                                kv=None,
                                                nuk=k,
                                                plot=None,
                                                kalone=enka,
                                                verbose='low')
        with open(root + 'nlttkcepstral.out', 'a') as f:
            f.write('{}\t'.format(kpoint * 10) + '{}\t'.format(np.real(tr)) + '{}\n'.format(np.real(ertr)))
        kpoints.append(kpoint)
        wcepstrum.append(tr)
        swcepstrum.append(ertr)
    return np.array(kpoints), np.array(wcepstrum), np.array(swcepstrum)

def computenlttcepstro_parallel(root, Np, L, nk, nkk, cp, deltat, tdump, nskip=1, enkainp=None, enkadata=None, ncpus=2):
    if enkainp==None:
        try:
            enk = np.load(root + 'enka.npy').T
            chk = np.load(root + 'chka.npy').T
            n1k = np.load(root + 'n1ka.npy').T
            n2k = np.load(root + 'n2ka.npy').T
        except:
            with open(root + 'enk.pkl', 'rb') as g:
                enkb = pk.load(g)
                enk = np.array(enkb).T
                enkb = 0
            with open(root + 'chk.pkl', 'rb') as g:
                chkb = pk.load(g)
                chk = np.array(chkb).T
                chkb = 0
            with open(root + 'n1k.pkl', 'rb') as g:
                n1kb = pk.load(g)
                n1k = np.array(n1kb).T
                n1kb = 0
            with open(root + 'n2k.pkl', 'rb') as g:
                n2kb = pk.load(g)
                n2k = np.array(n2kb).T
                n2kb = 0
        enka = (enk - enk[0,:, np.newaxis]/Np*(n1k+n2k)).T
    else:
        enka = enkadata

    print('start loop up to nkk')
    with Pool(ncpus) as p:
        inputs = [(root, Np, L, nk, cp, deltat, tdump, None, i, None, enka, 'low',) for i in range(1, nkk, nskip)]
        result = p.starmap(computenlttcepstro_k, inputs)

        return np.array(result)

def computenlttcepstro_k(root, Np, L, nk, cp, deltat, tdump, kv=None, nuk=None, plot=None, kalone=None, verbose='high'):
    print('START CALCULATION OF D(K)')
    if kalone is None:
        try:
            enk = np.load(root + 'enka.npy').T
            chk = np.load(root + 'chka.npy').T
            n1k = np.load(root + 'n1ka.npy').T
            n2k = np.load(root + 'n2ka.npy').T
        except:
            with open(root + 'enk.pkl', 'rb') as g:
                enkb = pk.load(g)
                enk = np.array(enkb).T
                enkb = 0
            with open(root + 'chk.pkl', 'rb') as g:
                chkb = pk.load(g)
                chk = np.array(chkb).T
                chkb = 0
            with open(root + 'n1k.pkl', 'rb') as g:
                n1kb = pk.load(g)
                n1k = np.array(n1kb).T
                n1kb = 0
            with open(root + 'n2k.pkl', 'rb') as g:
                n2kb = pk.load(g)
                n2k = np.array(n2kb).T
                n2kb = 0
        enka = (enk - enk[0,:, np.newaxis]/Np*(n1k+n2k)).T
        if verbose=='high':print('list transformed in np.array')
    else:
        enka = kalone
    rho = Np / (6.022e23 * L[0] * L[1] * L[2] * 1.e-30)  # mol/m^3

    fac = rho * cp  # J/k/m^3

    dt = deltat * tdump  # ps
    Gmod = tools.Ggeneratemodall(nk, L)

    if not kv:
        if verbose=='high':print('find k from number')
    else:
        klist=np.where(np.pi*2*Gmod==kv)
        k=klist[0]
    if not nuk:
        if verbose=='high': print('find k from value')
    else:
        k = nuk
    if (not kv) and (not nuk):

        raise ValueError('dimmi che k vuoi fare')


    start = time.time()
    f = open(root + 'enk{}.dat'.format(k), 'w')
    f.write('c_enk[1] c_enk[2] \n')
    for i in enka[:, k]:
        f.write('{} {}\n'.format(np.real(i), np.imag(i)))
    f.close()


    if verbose=='high': print('leggo file con i dati della densita` di energia')
    jfile = st.i_o.TableFile(root + 'enk{}.dat'.format(k), group_vectors=True)

    jfile.read_datalines(start_step=0, NSTEPS=0, select_ckeys=['enk'])

    DT_FS = 1

    if verbose=='high': print('calcolo l` oggetto corrente con sportran')

    j = st.Current([jfile.data['enk']], DT_FS=DT_FS,
                   KAPPA_SCALE=1)

    if verbose=='high': print('la frequenza di Nynquist e`  ', j.Nyquist_f_THz )

    fstar_THz = 4#�j.Nyquist_f_THz / 40

    if verbose=='high': print('resalmplo il periodogramma fino a fstar: ',  fstar_THz)

    jf = j.resample(fstar_THz=fstar_THz)

    if verbose=='high': print('Faccio l`analisi cepstrale sul periodogramma filtrato')

    jf.cepstral_analysis()

    if verbose=='high': print('calcolo l`autocorrelazione temporale della densita` di energia, '+
          'sto considerando parte reale e immaginaria come flussi indipendenti')

    jf.filter_psd(PSD_FILTER_W=0.5, freq_units='THz')
    #print(jf.fpsd[0]/2)
    j.compute_acf()

    chi = 0.5 * np.real(enka[:, k]) ** 2 + 0.5 * np.imag(enka[:, k]) ** 2 - np.mean(
        0.5 * np.real(enka[:, k]) + 0.5 * np.imag(enka[:, k])) ** 2

    v, b = tools.stdblock(chi)

    errrelchi=np.sqrt(v[int(19/20*len(v))])/(np.mean(chi))

    tr = np.pi*j.acfm[0] / (jf.cepf.tau_cutoffK * (2 * Gmod[k] * np.pi) ** 2) * fac / dt * (
        1e-10) ** 2 / 1e-12
    #print(np.mean(chi), j.acfm[0])
    #print((jf.cepf.tau_std_cutoffK / jf.cepf.tau_cutoffK * tr), (errrelchi*tr))

    ertr=np.pi*np.sqrt((jf.cepf.tau_std_cutoffK / jf.cepf.tau_cutoffK * tr)**2+(errrelchi*tr)**2)

    print('kpoint ', k,
          '\nkvalue ', 2 * Gmod[k] * np.pi *10, '1/nm',
          '\nD(k)', tr, 'W/mK',
          '\nstd D(k)', ertr, 'W/mK')

    os.remove(root + 'enk{}.dat'.format(k))
    print('DONE CALCULATION OF D(K), elapsed time:', -start + time.time())

    if plot:
        return j, jf, 2 * Gmod[k] * np.pi, tr, ertr
    else:
        return 2 * Gmod[k] * np.pi, tr, ertr



