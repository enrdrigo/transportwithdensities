import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from numba import njit, jit
from modules import initialize
from multiprocessing import Pool
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


def msd(dump, pos0, lenght):
    dist1 = np.zeros(lenght)
    dist2 = np.zeros(lenght)
    dist = np.zeros(lenght)
    snap = list(dump.keys())
    nsnap = len(snap)
    for i in range(lenght):
        j = i * int(len(snap) / lenght) + 2
        dumpdata = dump[str(j)][()].T
        list1 = np.where(dumpdata[1] == 1.)
        list2 = np.where(dumpdata[1] == 2.)
        pos = dumpdata[2:5]
        dist1[i] = np.linalg.norm(pos - pos0, axis=0)[list1].mean()
        dist2[i] = np.linalg.norm(pos - pos0, axis=0)[list2].mean()
        dist[i] = np.linalg.norm(pos - pos0, axis=0).mean()
    return dist1, dist2, dist


def gor(dump, lenght, L, ngridd):
    r = np.linspace(0.01, L[0] / 2, ngridd)
    ngrid = ngridd - 1
    gor1 = np.zeros(ngrid)
    gor2 = np.zeros(ngrid)
    gor12 = np.zeros(ngrid)
    gor1t = np.zeros((ngrid, lenght))
    gor2t = np.zeros((ngrid, lenght))
    gor12t = np.zeros((ngrid, lenght))
    snap = list(dump.keys())
    for i in range(lenght):
        j = i * int(len(snap) / lenght) + 1
        dumpdata = dump[str(j)][()].T
        list1 = np.where(dumpdata[1] == 1.)
        list2 = np.where(dumpdata[1] == 2.)
        pos = dumpdata[2:5]
        dist = (pos - pos.T[:, :, np.newaxis]).transpose(0, 2, 1)
        dist -= np.rint(dist / L) * L
        sqdist = np.linalg.norm(dist, axis=2)

        sqdist1 = sqdist[list1][:, list1][:, 0, :]
        sqdist1t = sqdist[list1][:, list1][:, 0, :]
        sqdist2 = sqdist[list2][:, list2][:, 0, :]
        sqdist2t = sqdist[list2][:, list2][:, 0, :]
        sqdist12 = sqdist[list1][:, list2][:, 0, :]
        sqdist12t = sqdist[list1][:, list2][:, 0, :]
        listr1 = []
        listr2 = []
        listr12 = []
        for t in range(ngrid):
            listr1g = np.where(sqdist1t < r[t + 1])
            listr1.append((len(sqdist1t[listr1g])))
            sqdist1t[listr1g] = 1.0e6

            listr2g = np.where(sqdist2t < r[t + 1])
            listr2.append((len(sqdist2t[listr2g])))
            sqdist2t[listr2g] = 1.0e6

            listr12g = np.where(sqdist12t < r[t + 1])
            listr12.append((len(sqdist12t[listr12g])))
            sqdist12t[listr12g] = 1.0e6

        gor1t[:, i] = np.array(listr1)
        gor2t[:, i] = np.array(listr2)
        gor12t[:, i] = np.array(listr12)

    return r[2:], gor1t.mean(axis=1) / r[1:] ** 2, gor2t.mean(axis=1) / r[1:] ** 2, gor12t.mean(axis=1) / r[1:] ** 2


def msdloop(dump, pos0, lenght, i):
    snap = len(dump)
    j = i * int(snap / lenght)
    dumpdata = dump[j].T
    list1 = np.where(dumpdata[1] == 1.)
    list2 = np.where(dumpdata[1] == 2.)
    pos = dumpdata[2:5]
    dist1 = (np.linalg.norm(pos - pos0, axis=0)[list1]**2).mean()
    dist2 = (np.linalg.norm(pos - pos0, axis=0)[list2]**2).mean()
    dist = (np.linalg.norm(pos - pos0, axis=0)**2).mean()
    return np.array([dist1, dist2, dist])


def msd_parallel(root='./', nblockstraj=10, timeskip=10, ncpus=os.cpu_count()):
    print('ncpus machine', os.cpu_count())
    with h5py.File(root + 'dump.h5', 'r') as dump:
        listasnap = []
        nsnap = list(dump.keys())
        nsnapperblock = int(len(nsnap) / nblockstraj)
        lenght = int(len(nsnap) / timeskip)
        lenghtperblock = int(lenght / nblockstraj)
        skipperblock = int(nsnapperblock / lenghtperblock)
        results = []
        for nread in range(nblockstraj):
            print(nread)

            listasnap = []
            startread = time.time()
            for i in range(1, nsnapperblock - 1, skipperblock):
                j = nread * nsnapperblock + i + 1
                listasnap.append(dump[str(j)][()])
            lenght = int(len(nsnap) / nblockstraj)
            print('read in ', time.time() - startread)
            startloop = time.time()
            with Pool(ncpus) as p:
                dumpdata = dump[str(1)][()].T
                pos0 = dumpdata[2:5]
                inputs = [(listasnap, pos0, lenghtperblock, i) for i in range(lenghtperblock)]
                result = p.starmap(msdloop, inputs)

            if nread == 0:
                results.extend(result)
            else:
                results.extend(result)
            print('compute in ', time.time() - startloop)
    return np.array(results).T


def gorloop(dump, L, ngrid, lenght, r, i):
    snap = len(dump)
    gor1t = np.zeros(ngrid)
    gor2t = np.zeros(ngrid)
    gor12t = np.zeros(ngrid)
    j = i * int(snap / lenght)
    dumpdata = dump[j].T
    list1 = np.where(dumpdata[1] == 1.)
    list2 = np.where(dumpdata[1] == 2.)
    pos = dumpdata[2:5]
    dist = (pos - pos.T[:, :, np.newaxis]).transpose(0, 2, 1)
    dist -= np.rint(dist / L) * L
    sqdist = np.linalg.norm(dist, axis=2)

    sqdist1 = sqdist[list1][:, list1][:, 0, :]
    sqdist1t = sqdist[list1][:, list1][:, 0, :]
    sqdist2 = sqdist[list2][:, list2][:, 0, :]
    sqdist2t = sqdist[list2][:, list2][:, 0, :]
    sqdist12 = sqdist[list1][:, list2][:, 0, :]
    sqdist12t = sqdist[list1][:, list2][:, 0, :]
    listr1 = []
    listr2 = []
    listr12 = []
    for t in range(ngrid):
        listr1g = np.where(sqdist1t < r[t + 1])
        listr1.append((len(sqdist1t[listr1g])))
        sqdist1t[listr1g] = 1.0e6

        listr2g = np.where(sqdist2t < r[t + 1])
        listr2.append((len(sqdist2t[listr2g])))
        sqdist2t[listr2g] = 1.0e6

        listr12g = np.where(sqdist12t < r[t + 1])
        listr12.append((len(sqdist12t[listr12g])))
        sqdist12t[listr12g] = 1.0e6

    gor1t = np.array(listr1)
    gor2t = np.array(listr2)
    gor12t = np.array(listr12)
    return np.array([gor1t, gor2t, gor12t])


def gor_parallel(root='./', filename='dump.lammpstrj', nblockstraj=10, timeskip=10, ngridd=100, ncpus=os.cpu_count(), Lgr=0):
    print('ncpus machine', os.cpu_count())

    L, Linf = initialize.getBoxboundary(filename, root)
    if Lgr==0:
        rmax = L[0] / 2
    else:
        rmax=Lgr
    r = np.linspace(0.01, rmax, ngridd)
    ngrid = ngridd - 1
    with h5py.File(root + 'dump.h5', 'r') as dump:
        listasnap = []
        nsnap = list(dump.keys())
        nsnapperblock = int(len(nsnap) / nblockstraj)
        lenght = int(len(nsnap) / timeskip)
        lenghtperblock = int(lenght / nblockstraj)
        skipperblock = int(nsnapperblock / lenghtperblock)
        results = []
        for nread in range(nblockstraj):
            print(nread)

            listasnap = []
            startread = time.time()
            for i in range(1, nsnapperblock - 1, skipperblock):
                j = nread * nsnapperblock + i + 1
                listasnap.append(dump[str(j)][()])
            lenght = int(len(nsnap) / nblockstraj)
            print('read in ', time.time() - startread)
            startloop = time.time()
            with Pool(ncpus) as p:
                inputs = [(listasnap, L, ngrid, lenghtperblock, r, i) for i in range(lenghtperblock)]
                result = p.starmap(gorloop, inputs)

            if nread == 0:
                results.extend(result)
            else:
                results.extend(result)
            print('compute in ', time.time() - startloop)
        return r[2:], np.array(results).mean(axis=0)[:, 1:] / r[2:] ** 2