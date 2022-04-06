import numpy as np
import pickle as pk
from numba import njit
import time
import h5py
import os
import json


def read_dump(root, filename, Np, ntry):
    with open(root + filename, 'r') as f:
        print('ok')

        if os.path.exists(root + 'dump.h5'):
            with h5py.File('dump.h5', 'r') as dump:
                snap = list(dump.keys())

            lenght = len(snap)
            print('THE LOADING WAS STOPPED AT THE SNAPSHOT: ', lenght)

        else:
            dump = h5py.File('dump.h5', 'a')
            lenght = 0

        dump = h5py.File(root + 'dump.h5', 'a')
        d = []
        start = time.time()

        print(f.name)

        for index, line in enumerate(f):

            if index < lenght * (Np + 9):
                continue

            linesplit = line.split(' ')

            if len(linesplit) != 7 and len(linesplit) != 8:
                continue
            dlist = [float(linesplit[i]) for i in range(7)]
            d.append(dlist)

            if (index + 1) % (Np + 9) == 0:

                if len(d) == 0:
                    print('END READ FILE')
                    print('got ' + str((index + 1) // (Np + 9)) + ' snapshot')
                    dump.close()
                    return

                elif len(d) != Np:

                    print(len(d))
                    print('STOP: THE SNAPSHOT ' + str((index + 1) // (Np + 9)) + ' DOES NOT HAVE ALL THE PARTICLES')
                    print('got ' + '' + ' snapshot')
                    dump.close()
                    return

                datisnap = np.array(d)
                d = []
                dump.create_dataset(str((index + 1) // (Np + 9)),
                                    data=datisnap)  # compression for float do not work well
                # print(index, (index + 1) / (Np + 9))

                if (index + 1) // (Np + 9) + 3 == ntry * 3:
                    print('number of total snapshots is', (index + 1) // (Np + 9) / 3)
                    print('done')
                    print('END READ. NO MORE DATA TO LOAD. SEE NTRY')
                    dump.close()
                    return

                if (index + 1) // (Np + 9) + 3 == ntry * 3:
                    print('number of total snapshots is', (index + 1) // (Np + 9))
                    print('done')
                    print('elapsed time: ', time.time() - start)
                    print('END READ NTRY')
                    dump.close()
                    return

        print('number of total snapshots is', (index + 1) // (Np + 9))
        print('done')
        print('elapsed time: ', time.time() - start)
        print('END READ FILE GOOD')
        dump.close()
        return


def molar(root, Np):
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
