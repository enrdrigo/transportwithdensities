import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os


def stdblock(array):
    var = list()
    binsize = list()
    nbino = 0
    for i in range(1, int(len(array) / 10)):
        nbin = int(len(array) / i)
        if nbin == nbino:
            continue
        rarray = np.reshape(array[:nbin * i], (nbin, i))
        barray = np.zeros(nbin, dtype=np.complex_)
        barray = np.sum(rarray, axis=1) / i
        var.append(np.var(barray) / nbin)
        binsize.append(i)
        nbino = nbin

    return np.array(var), np.array(binsize)

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
    G[0] = np.array([0, 0, 0]) / L + 2.335581758729501e-06 / 2 / np.pi / np.sqrt(3.0)
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


def computestaticresponse(root, L, nk, temp):
    plot = False

    if os.path.exists(root + 'chk.pkl'):
        with open(root + 'enk.pkl', 'rb') as g:
            enkb = pk.load(g)
            enk = np.transpose(np.array(enkb))
        with open(root + 'chk.pkl', 'rb') as g:
            chkb = pk.load(g)
            chk = np.transpose(np.array(chkb))
        nsnap = int(len(chkb))
    else:
        raise ValueError

    fac = (16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / (L[0] * L[1] * L[2] * 1.0e-30 * 1.38e-23 * temp ** 2 * 8.854 * 1.0e-12))
    face = (16.022 ** 2) * 1.0e5 / (L[0] * L[1] * L[2] * 1.38 * temp * 8.854)

    xk = Ggeneratemodall(nk, L) * 2 * np.pi

    # xk = np.linspace(0, nk - 1, nk) * 2 * np.pi / L + np.sqrt(3.) * 1.0e-5 * 2 * np.pi / L

    a = np.zeros(nk, np.complex_)
    b = np.zeros(nk, np.complex_)
    c = np.zeros(nk, np.complex_)
    d = np.zeros(nk, np.complex_)
    e = np.zeros(nk, np.complex_)

    va = np.zeros(nk)
    vb = np.zeros(nk)
    vc = np.zeros(nk)
    vd = np.zeros(nk)
    ve = np.zeros(nk)

    with open(root + 'enk.out', '+w') as g:
        for i in range(nk):
            g.write('{}\t'.format(xk[i]) + '{}\n'.format(np.abs(np.mean(enk[i]))))

    for i in range(nk):
        a[i] = np.mean((enk[i] / xk[i]) * np.conj(chk[i] / xk[i])) * fac
        d[i] = np.mean((chk[i] / xk[i]) * np.conj(chk[i] / xk[i])) * face

    convergence1 = np.real((np.cumsum((enk[0][:] / xk[0]) * np.conj(chk[0][:] / xk[0])) * fac) / (
            np.cumsum((chk[0][:] / xk[0]) * np.conj(chk[0][:] / xk[0])) * face))
    convergence2 = np.real((np.cumsum((enk[1][:] / xk[1]) * np.conj(chk[1][:] / xk[1])) * fac) / (
            np.cumsum((chk[1][:] / xk[1]) * np.conj(chk[1][:] / xk[1])) * face))
    convergence3 = np.real((np.cumsum((enk[2][:] / xk[2]) * np.conj(chk[2][:] / xk[2])) * fac) / (
            np.cumsum((chk[2][:] / xk[2]) * np.conj(chk[2][:] / xk[2])) * face))
    convergence4 = np.real((np.cumsum((enk[3][:] / xk[3]) * np.conj(chk[3][:] / xk[3])) * fac) / (
            np.cumsum((chk[3][:] / xk[3]) * np.conj(chk[3][:] / xk[3])) * face))

    with open(root + 'convergence.out', '+w') as g:
        for i in range(1, len(enk[0]), 10):
            g.write('{}\t'.format(i) + '{}\t'.format(convergence1[i]) + '{}\t'.format(convergence2[i]) + '{}\t'.format(
                convergence3[i]) + '{}\n'.format(convergence4[i]))

    for i in range(nk):
        std, bins = np.sqrt(stdblock((enk[i] / xk[i]) * np.conj(chk[i] / xk[i]) * fac))
        pp = int(19 * len(std) / 20)
        va[i] = std[pp]
        std, bins = np.sqrt(stdblock((chk[i] / xk[i]) * np.conj(chk[i] / xk[i]) * face))
        vd[i] = std[pp]

    with open(root + 'staticresponse.out', '+w') as g:
        g.write('#k\t chtpc\t chdiel\n')
        for i in range(nk):
            g.write('{} \t'.format(xk[i]))
            g.write('{} \t'.format(np.real(a[i])) + '{} \t'.format(np.real(va[i])))
            g.write('{} \t'.format(np.real(d[i])) + '{} \n'.format(np.real(vd[i])))

    v, x = stdblock((chk[1] / xk[1]) * np.conj(chk[1] / xk[1]) * face)
    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.plot(x, np.sqrt(v))
        plt.ylabel(r'$\sigma_b$ of $\langle\frac{\rho(k_{min})\rho(-k_{min})}{k_{min}^2}\rangle$')
        plt.xlabel('block size')
        plt.show(block=False)

    with open(root + 'blockanalisisvardckmin.out', 'w+') as g:
        for i in range(len(v)):
            g.write('{}\t'.format(x[i]) + '{}\n'.format(np.sqrt(v[i])))

    v, x = stdblock((enk[1] / xk[1]) * np.conj(chk[1] / xk[1]) * fac)
    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.plot(x, np.sqrt(v))
        plt.ylabel(r'$\sigma_b$ of $\langle\frac{\rho(k_{min})\left(e(-k_{min})-e(0)\right)}{k_{min}^2}\rangle$')
        plt.xlabel('block size')
        plt.show(block=False)

    with open(root + 'blockanalisisvartpckmin.out', 'w+') as g:
        for i in range(len(v)):
            g.write('{}\t'.format(x[i]) + '{}\n'.format(np.sqrt(v[i])))

    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.errorbar(xk[0:], d[0:], vd, fmt='.-', label=r'$\langle\frac{\rho(k)\rho(-k)}{k^2}\rangle$')
        plt.xlabel(r'k ($\AA^{-1}$)')
        plt.ylabel(r'$\epsilon_r$')
        plt.legend()
        plt.show(block=False)

        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.errorbar(xk[0:], a[0:], va, fmt='.-', label=r'$\langle\frac{\rho(k)\left(e(-k)-e(0)\right)}{k^2}\rangle$')
        plt.xlabel(r'k ($\AA^{-1}$)')
        plt.ylabel(r'$\frac{P}{\epsilon_0\triangledown (T)/T }$ (V)')
        plt.legend()
        plt.show(block=False)

    stdch = np.real(np.sqrt((va / (1 - 1 / d[0]) / d[0] ) ** 2))
    tpcch = np.real(a / (1 - 1 / d[0]) / d[0] )
    # divido la risposta in polarizzazione al gradiente di temperature per il valore asintotica della risposta in P a D:
    # cosi' ho la risposta in D al gradiente di T. A questo punto divido ancora per la costante dielettrica: E/grad(T)!
    print('relative dielectric constant dipoles:', d[0])
    print('relative dielectric contant charges k_min:', 1/(1-d[1]))

    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.errorbar(xk, tpcch, stdch, fmt='.-', label='computed via the charges')
        plt.xlabel(r'k ($\AA^{-1}$)')
        plt.ylabel(r'$\frac{E}{\triangledown (T) }$ (V/K)')
        plt.legend()
        plt.show(block=False)

    with open(root + 'thermopolarizationresponse.out', '+w') as g:
        g.write('# k\t tpc via the charge \n')
        for i in range(1, nk):
            g.write('{}\t'.format(xk[i]))
            g.write('{}\t'.format(tpcch[i]) + '{}\n'.format(stdch[i]))
    if plot:
        plt.show()

    out = dict()

    out['dielectric'] = dict()

    out['thermopolarization'] = dict()

    out['dielectric']['charge'] = {'mean': d, 'std': vd}

    out['thermopolarization']['charge'] = {'mean': a, 'std': va}

    return out

