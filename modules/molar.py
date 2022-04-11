import numpy as np
import time
import h5py
import os
from modules import initialize

def molar_old(root, Np):
    start = time.time()
    print(
        'la routine legge il file dump.h5 con posizioni SCALATE e le energie per atomo.\n Calcola i volumi parziali (un unita` del volume per particella) e le energie parziali.')
    if os.path.exists(root + 'dump.h5'):
        pass
    else:
        raise ValueError('crea il file dump.h5!! con la routine read_dump')
    with h5py.File(root + 'dump.h5', 'r') as dump:

        snap = list(dump.keys())
        energies = np.zeros((len(snap), 3))
        enmean = np.zeros(len(snap))
        N1 = np.zeros((len(snap), 3))
        N2 = np.zeros((len(snap), 3))
        fetta = {'x': 2, 'y': 3, 'z': 4}
        portions = ['x', 'y', 'z']

        for i in range(1, len(snap) + 1):

            j = i - 1
            dumpdata = dump[str(i)][()].T
            enmean[j] = dumpdata[5].sum() + dumpdata[6].sum()

            for s in portions:
                sin = portions.index(s)
                list_fetta = np.where(dumpdata[fetta[s]] < 0.5)
                energies[j, sin] = dumpdata[5][list_fetta].sum() + dumpdata[6][list_fetta].sum()
                list_fetta_sp1 = list_fetta[0][np.where(dumpdata[1][list_fetta] == 1.)]
                list_fetta_sp2 = list_fetta[0][np.where(dumpdata[1][list_fetta] == 2.)]
                N1[j, sin] = len(list_fetta_sp1)
                N2[j, sin] = len(list_fetta_sp2)

    n1 = N1.mean(axis=0)
    n2 = N2.mean(axis=0)
    energy = energies.mean(axis=0)
    delta = np.einsum('imn,lmn->ilmn', \
                      np.array([N1 - n1, N2 - n2]), \
                      np.array([N1 - n1, N2 - n2])).mean(axis=2)

    x = np.array([n1 / (n1 + n2), n2 / (n1 + n2)]).transpose((1, 0))
    alpha = np.einsum('lm,ilm->ilm', \
                      energies - energy, \
                      np.array([N1 - n1, N2 - n2])).mean(axis=1).transpose((1, 0))

    dinv = np.linalg.inv(delta.transpose((2, 0, 1)))
    den = np.einsum('ij,ik,ijk->i', \
                    x, \
                    x, \
                    dinv) ** -1

    # partial molar volumes, in unita' di volume per particella
    v = np.einsum('il,i->il', \
                  np.einsum('ilm,il->im', \
                            dinv, \
                            x), \
                  den)

    # partial molar energies
    u = np.einsum('i,il->il', \
                  energy / (n1 + n2), \
                  v) + \
        np.einsum('ab,abc->ac', \
                  alpha, \
                  dinv - np.einsum('abc,a->abc', \
                                   np.einsum('ab,abc,ad,adf->acf', \
                                             x, \
                                             dinv, \
                                             x, \
                                             dinv), \
                                   den))

    print('volumi parizali', v.mean(axis=0))
    print('energie parziali', u.mean(axis=0))
    print('relazione di eulero per l`energia parziale', enmean.mean() / Np, u.mean(axis=0).mean())
    print('elapsed time: ', time.time() - start)
    return v, u


def molar(root, Np):
    start = time.time()
    print('La routine legge il file dump.h5 con posizioni NON SCALATE e le energie per atomo.\n ' +
          'Calcola i volumi parziali (un unita` del volume per particella) e le energie parziali.')
    if os.path.exists(root + 'dump.h5'):
        pass
    else:
        raise ValueError('crea il file dump.h5!! con la routine read_dump')

    L, L_min = initialize.getBoxboundary('dump1.1fs.lammpstrj', root)

    with h5py.File(root + 'dump.h5', 'r') as dump:

        snap = list(dump.keys())
        energies = np.zeros((len(snap), 3))
        enmean = np.zeros(len(snap))
        N1 = np.zeros((len(snap), 3))
        N2 = np.zeros((len(snap), 3))
        fetta = {'x': 0, 'y': 1, 'z': 2}
        portions = ['x', 'y', 'z']

        for i in range(1, len(snap) + 1):

            j = i - 1
            dumpdata = dump[str(i)][()].T
            enmean[j] = dumpdata[5].sum() + dumpdata[6].sum()
            pos = wrappos(dumpdata[2:5], L, L_min)
            posunw = dumpdata[2:5]

            for s in portions:
                sin = portions.index(s)
                list_fetta = np.where(pos[fetta[s]] < 0.5)
                energies[j, sin] = dumpdata[6][list_fetta].sum() + dumpdata[7][list_fetta].sum()
                list_fetta_sp1 = list_fetta[0][np.where(dumpdata[1][list_fetta] == 1.)]
                list_fetta_sp2 = list_fetta[0][np.where(dumpdata[1][list_fetta] == 2.)]
                N1[j, sin] = len(list_fetta_sp1)
                N2[j, sin] = len(list_fetta_sp2)

    n1 = N1.mean(axis=0)
    n2 = N2.mean(axis=0)
    energy = energies.mean(axis=0)
    delta = np.einsum('imn,lmn->ilmn', \
                      np.array([N1 - n1, N2 - n2]), \
                      np.array([N1 - n1, N2 - n2])).mean(axis=2)

    x = np.array([n1 / (n1 + n2), n2 / (n1 + n2)]).transpose((1, 0))
    alpha = np.einsum('lm,ilm->ilm', \
                      energies - energy, \
                      np.array([N1 - n1, N2 - n2])).mean(axis=1).transpose((1, 0))

    dinv = np.linalg.inv(delta.transpose((2, 0, 1)))
    den = np.einsum('ij,ik,ijk->i', \
                    x, \
                    x, \
                    dinv) ** -1

    # partial molar volumes, in unita' di volume per particella
    v = np.einsum('il,i->il', \
                  np.einsum('ilm,il->im', \
                            dinv, \
                            x), \
                  den)

    # partial molar energies
    u = np.einsum('i,il->il', \
                  energy / (n1 + n2), \
                  v) + \
        np.einsum('ab,abc->ac', \
                  alpha, \
                  dinv - np.einsum('abc,a->abc', \
                                   np.einsum('ab,abc,ad,adf->acf', \
                                             x, \
                                             dinv, \
                                             x, \
                                             dinv), \
                                   den))

    print('volumi parizali', v.mean(axis=0), np.sum(v.mean(axis=0) * x.mean(axis=0)))
    print('energie parziali', u.mean(axis=0))
    print('relazione di eulero per l`energia parziale', enmean.mean() / Np, np.sum(u.mean(axis=0) * x.mean(axis=0)))
    print('elapsed time: ', time.time() - start)
    return v, u


def wrappos(posunw, L, L_min):
    return (np.mod((posunw.T - L_min), L) / L).T


def read_log_lammps(root):
    start = time.time()
    print(
        'Inizio la lettura del file log.lammps per le quantita` globali come energia totale, pressione temperatura \n o entalpia. In output verra scritto un file python ' + root + 'log.lammps.npy' + ' con un dizionario')
    datadic = {}
    with open(root + 'log.lammps', 'r') as f:

        startcollect = False

        for index, line in enumerate(f):

            linesplit = []

            for i in line.split(' '):

                if i != '': linesplit.append(i)

            if linesplit[0] == 'thermo_style':
                dickeys = [str(i) for i in linesplit[2:]]
                data = [[] for i in dickeys]

            if linesplit[0] == 'Step': startcollect = True; continue
            if linesplit[0] == 'Loop': break
            if startcollect:

                for i in range(len(dickeys)):
                    data[i].append(float(linesplit[i]))

        datadic = {dickeys[0]: np.array(data[0])}

        for i in dickeys[1:]:
            datadic[i] = np.array(data[dickeys.index(i)])
    print('elapsed time: ', time.time() - start)
    print(type(datadic))
    np.save(root + 'log.lammps.npy', datadic)
    return datadic


def molar_enthalpy(root, volume, Np):
    volumepp = volume / Np
    if os.path.exists(root + 'log.lammps.npy'):
        pass
    else:
        raise ValueError('crea il file log.lammps.npy!! con la routine read_log_lammps')
    dic_data_log = np.load(root + 'log.lammps.npy', allow_pickle='TRUE').item()

    dic_data_log.keys()

    v, u = molar(root, Np)

    h = u.mean(axis=0) + np.mean(dic_data_log['press']) * volumepp * v.mean(axis=0)
    return h
