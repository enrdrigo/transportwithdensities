import pickle as pk
import numpy as np
from scipy import signal
import os


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

def computenltt(root, Np, L, nk, cp, deltat, tdump):
    if os.path.exists(root + 'chk.pkl'):
        with open(root + 'enk.pkl', 'rb') as g:
            enkb = pk.load(g)
            enk = np.transpose(np.array(enkb))
        with open(root + 'chk.pkl', 'rb') as g:
            chkb = pk.load(g)
            chk = np.transpose(np.array(chkb))
        nsnap = int(len(enkb)/3)
    else:
        raise ValueError

    # nsnap, enk, dipenkx, dipenky, chk, dipkx, dipky = computekft(root, filename, Np, L, posox, nk, ntry, natpermol)

    ndata = int(enk.shape[1])

    enkcorr = enk  # np.reshape(enk, (nk, int(ndata)))

    nblocks = 10

    tblock = int(enkcorr.shape[1] / nblocks)

    tinblock = int(tblock / 2)

    rho = Np / (6.022e23 * L[0] * L[1] * L[2] * 1.e-30)  # mol/m^3

    fac = rho * cp  # J/k/m^3

    dt = deltat * tdump  # ps

    corr = np.zeros((nblocks, tinblock), dtype=np.complex_)

    chi = np.var(np.real(enk[:, :]), axis=1)  # (eV**2

    corren = np.zeros((nblocks, nk - 1, int(tinblock / 2) + 1), dtype=np.complex_)

    ft = np.zeros((nk - 1, int(tinblock / 2) + 1), dtype=np.complex_)

    corrk = np.zeros((nblocks, nk-1, tinblock), dtype=np.complex_)

    Gmod = Ggeneratemodall(nk, L)

    for t in range(nblocks):

        print(t)

        for j in range(1, nk):
            corr = np.zeros((nblocks, tinblock), dtype=np.complex_)
            for i in range(0, tinblock, int(tinblock / 10)):
                corr[t] += (autocorr(enkcorr[j, (tblock * t +i):(tblock * t + tinblock +i)])) / 10

            chik = (np.var(np.real(enkcorr[j, :])))

            ft[j - 1] = chik / (np.cumsum(corr[t, :int(tinblock / 2) + 1]) * (2 * Gmod[j] * np.pi) ** 2) * (
                        fac / dt * (1e-10) ** 2 / 1e-12)
            corrk[t, j-1] = corr[t]
            #print(chik, corr[t,0])

        corren[t] = ft
    return corren,  chi, corrk

