import numpy as np
from numba import njit, prange
import time


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE STATIC DIELECTRIC CONSTANT (SDC) AT A GIVEN PHYSICAL G VECTOR. HERE WE RESTRICT OURSELVES IN THE
# DIRECTION (1,0,0). THE SDC IS CALCULATED VIA THE AVERAGE VALUE OF THE MODULUS SQUARED OF THE FOURIER TRANFORM OF THE
# POLARIZATION IN G OR VIA THE MODULUS SQUARED OF THE FOURIER TRANSORM OF THE CHARGE IN G DIVIDED BY THE MODULUS OF G.

@njit(fastmath=True, parallel=True)
def computestatdc(nk, dipmol, cdmol, chat, pos, L, nsnap):
    e0pol = np.zeros((nk, 3))
    e0ch = np.zeros(nk)
    for j in range(nk):
        e0pol[j][0] = (16.022**2) * np.sum(np.abs(np.sum(np.transpose(dipmol)[0] *\
                    np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0))**2)/nsnap\
                    * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
        e0pol[j][1] = (16.022**2) * np.sum(np.abs(np.sum(np.transpose(dipmol)[1] *\
                    np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0))**2)/nsnap\
                    * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
        e0pol[j][2] = (16.022**2) * np.sum(np.abs(np.sum(np.transpose(dipmol)[2] *\
                    np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0))**2)/nsnap\
                    * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
        e0ch[j] = (16.022**2) * np.mean(np.abs(np.sum(np.transpose(chat) *\
                np.exp(1j * (np.transpose(pos)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0))**2)\
                / ((j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8)**2) * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
    return e0pol, e0ch


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE STATIC THERMOPOLARIZATION COEFFICIENT (STPC) AT A GIVEN PHYSICAL G VECTOR. HERE WE RESTRICT OURSELVES IN THE
# DIRECTION (1,0,0). WE MUST DIVIDE BY THE STATIC RESPONSE (LONGITUDINAL).
@njit(fastmath=True, parallel=True)
def thermopolcoeff(nk, chat, enat, em, pos, posatomic, L, nsnap):
    tpc = np.zeros(nk, dtype=np.complex_)
    sdtpc = np.zeros(nk)
    for j in range(nk):
        tpc[j] = np.mean(np.sum(np.transpose(chat)*np.exp(1j * (np.transpose(pos)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0) *\
                (np.sum(np.transpose(enat)*np.exp(-1j * (np.transpose(posatomic)[0] * (j * 2 * np.pi / L + + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0)-em))\
                 / ((j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8)**2) * \
                 16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / ((L) ** 3 * 1.0e-30 * 1.38e-23 * 300 * 300 * 8.854 * 1.0e-12)
        sdtpc[j] = np.sqrt(np.var(np.sum(np.transpose(chat)*np.exp(1j * (np.transpose(pos)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0) *\
                (np.sum(np.transpose(enat)*np.exp(-1j * (np.transpose(posatomic)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0)-em)))\
                 / ((j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8)**2) * \
                 16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / ((L) ** 3 * 1.0e-30 * 1.38e-23 * 300 * 300 * 8.854 * 1.0e-12)/nsnap

    return tpc, sdtpc

@njit(fastmath=True, parallel=True)
def thermopoldipcoeff(nk, dipmol, endip, cdmol,  L, nsnap):
    tpcdip = np.zeros((nk, 3), dtype=np.complex_)
    for j in range(nk):
        tpcdip[j][0] = np.mean(np.sum(np.transpose(dipmol)[0] * np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0) *\
                (np.sum(np.transpose(endip)[0]*np.exp(-1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0)))* \
                 16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / ((L) ** 3 * 1.0e-30 * 1.38e-23 * 300 * 300 * 8.854 * 1.0e-12)

        tpcdip[j][1] = np.mean(np.sum(np.transpose(dipmol)[1] * np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))),axis=0) * \
                (np.sum(np.transpose(endip)[1] * np.exp(-1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0))) * \
                16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / ((L) ** 3 * 1.0e-30 * 1.38e-23 * 300 * 300 * 8.854 * 1.0e-12)

        tpcdip[j][2] = np.mean(np.sum(np.transpose(dipmol)[2] * np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))),axis=0) * \
                (np.sum(np.transpose(endip)[2] * np.exp(-1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8))), axis=0))) * \
                16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / ((L) ** 3 * 1.0e-30 * 1.38e-23 * 300 * 300 * 8.854 * 1.0e-12)

    return tpcdip


# ----------------------------------------------------------------------------------------------------------------------

def reshape(cdmol, dipmol):
    nmol=np.shape(dipmol)[1]
    nsnap=np.shape(dipmol)[0]
    # DIPMOL, CDMOL HAVE DIMENTION (NSNAP, NMOL, 3)
    rcdmol=np.zeros((3, nmol, nsnap))
    rcdmol = np.transpose(cdmol, (2, 1, 0))
    rdipmol = np.transpose(dipmol, (2, 1, 0))
    tcdmol = np.transpose(cdmol, (2, 1, 0))
    tdipmol = np.transpose(dipmol, (2, 1, 0))
    return rdipmol, rcdmol, tdipmol, tcdmol


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE DIPOLE PAIR CORRELATION FUNCTION AND ITS STD

@njit(parallel=True, fastmath=True)
def dip_paircf(G, nk, rdipmol, rcdmol, tdipmol, tcdmol, L, nsnap):
    nmol = int(np.shape(rdipmol)[1])
    distcdm = np.zeros((nmol, nsnap))
    diffcdm = np.zeros((3, nmol, nsnap))
    dipsq = np.zeros((3, nmol, nsnap), dtype=np.complex_)
    gk = np.zeros((nk, 3))
    stdgk = np.zeros((nk, 3))
    sigma = 0.08
    cm = np.zeros((3, nmol, nsnap), dtype=np.complex_)
    # RCDMOL, RDIPMOL HAVE SIZE (3, NMOL, NSNAP), (3, NMOL, NSNAP)

    for i in range(nk):
        r = i * (L - 2) / 2 / nk + 2
        #start = time.time()

        for s in range(nmol):
            for c in range(3):
                diffcdm[c, :, :] = rcdmol[c, s, :] - tcdmol[c, :, :]

                dipsq[c, :, :] = rdipmol[c, s, :]*tdipmol[c, :, :] * np.exp(1j * diffcdm[0, :, :] * (G * 2 * np.pi / L))

                dipsq[c, :, s] = 0

            distcdm[:, :] = np.sqrt(np.sum(diffcdm*diffcdm, axis=0))
            for c in range(3):
                cm[c, s, :] = np.sum(dipsq[c, :, :]*np.exp(-(r-distcdm[:, :])**2/sigma**2), axis=0)

        gk[i] = np.real(np.sum(np.sum(cm, axis=1)/nmol, axis=1) / nsnap / r ** 2 / 2 * np.pi / sigma ** 2)
        stdgk[i][0] = np.sqrt(np.real(np.var(np.sum(cm, axis=1) / nmol, ) / nsnap)) / r ** 2 / 2 * np.pi / sigma ** 2
        stdgk[i][1] = np.sqrt(np.real(np.var(np.sum(cm, axis=1) / nmol, ) / nsnap)) / r ** 2 / 2 * np.pi / sigma ** 2
        stdgk[i][2] = np.sqrt(np.real(np.var(np.sum(cm, axis=1) / nmol, ) / nsnap)) / r ** 2 / 2 * np.pi / sigma ** 2

        print((i*(L - 2)/2/nk + 2)/0.529, gk[i][0], gk[i][1], gk[i][2], stdgk[i][0], stdgk[i][1], stdgk[i][2])

    return gk, stdgk




