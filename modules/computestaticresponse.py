import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os
from modules import tools


def computestaticresponse(root, L, nk, temp, plot=False):

    if os.path.exists(root + 'chk.pkl'):
        with open(root + 'chk.pkl', 'rb') as g:
            chkb = pk.load(g)
            chk = np.transpose(np.array(chkb))
        nsnap = int(len(chkb))
    else:
        raise ValueError('Compute the chk.pkl file con la routine traj.py!!')

    face = (16.022 ** 2) * 1.0e5 / (L[0] * L[1] * L[2] * 1.38 * temp * 8.854)
    xk = tools.Ggeneratemodall(nk, L) * 2 * np.pi

    d = np.zeros(nk, np.complex_)

    vd = np.zeros(nk)


    for i in range(nk):
        d[i] = np.mean((chk[i] / xk[i]) * np.conj(chk[i] / xk[i])) * face

    convergence1 = np.real(((chk[0][:] / xk[0]) * np.conj(chk[0][:] / xk[0])) * face)
    convergence2 = np.real(((chk[1][:] / xk[1]) * np.conj(chk[1][:] / xk[1])) * face)

    with open(root + 'convergence_dielectric.out', '+w') as g:
        for i in range(1, len(chk[0])):
            g.write('{}\t'.format(i) + '{}\t'.format(convergence1[i]) + '{}\n'.format(convergence2[i]))

    for i in range(nk):
        std, bins = np.sqrt(tools.stdblock((chk[i] / xk[i]) * np.conj(chk[i] / xk[i]) * face))
        pp = int(19 * len(std) / 20)
        vd[i] = std[pp]

    with open(root + 'staticresponse.out', '+w') as g:
        g.write('#k\t chtpc\t chdiel\n')
        for i in range(nk):
            g.write('{} \t'.format(xk[i]))
            g.write('{} \t'.format(np.real(d[i])) + '{} \n'.format(np.real(vd[i])))

    v, x = tools.stdblock((chk[1] / xk[1]) * np.conj(chk[1] / xk[1]) * face)
    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.plot(x, np.sqrt(v))
        plt.ylabel(r'$\sigma_b$ of $\langle\frac{\rho(k_{min})\rho(-k_{min})}{k_{min}^2}\rangle$')
        plt.xlabel('block size')
        plt.show(block=False)

    with open(root + 'blockanalisisvardckmin.out', 'w+') as g:
        for i in range(len(v)):
            g.write('{}\t'.format(x[i]) + '{}\n'.format(np.sqrt(v[i])))


    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.errorbar(xk[0:], d[0:], vd, fmt='.-', label=r'$\langle\frac{\rho(k)\rho(-k)}{k^2}\rangle$')
        plt.xlabel(r'k ($\AA^{-1}$)')
        plt.ylabel(r'$\epsilon_r$')
        plt.legend()
        plt.show(block=False)

    print('relative dielectric constant dipoles:', d[0])
    print('relative dielectric contant charges k_min:', 1/(1-d[1]))

    if plot:
        plt.show()

    out = dict()

    out['dielectric'] = dict()

    out['dielectric']['charge'] = {'mean': d, 'std': vd}


    return out

