import numpy as np
import pickle as pk
from modules import compute
from numba import njit
import time
import h5py
import os
import warnings


def read_dump(root, filename, Np, ntry):
    with open(root + filename, 'r') as f:
        for indexx, lines in enumerate(f):

            linesplit = []

            for i in lines.split(' '):

                if i != '': linesplit.append(i)

            if len(linesplit) > 2 and linesplit[1] == 'ATOMS':
                dickeys=[]
                for i in linesplit[2:]:
                    if i =='\n': continue
                    dickeys.append(str(i))
                print('you dumped this quantities:\n', dickeys)
                print('In order for the code to work I need ',
                      ['id', 'type', 'xu', 'yu', 'zu', 'q', 'c_enp', 'c_enk\n'])
                warnings.warn('modifica questa parte per renderla piu` generale')
                break

    with open(root + filename, 'r') as f:

        if os.path.exists(root + 'dump.h5'):
            with h5py.File(root + 'dump.h5', 'r') as dump:
                snap = list(dump.keys())

            lenght = len(snap)
            print('THE LOADING WAS STOPPED AT THE SNAPSHOT: ', lenght)

        else:
            dump = h5py.File(root + 'dump.h5', 'a')
            lenght = 0

        dump = h5py.File(root + 'dump.h5', 'a')
        d = []
        start = time.time()

        print(f.name)

        for index, line in enumerate(f):

            if index < lenght * (Np + 9):
                continue

            linesplit = []

            for i in line.split(' '):

                if i != '': linesplit.append(i)

            if len(linesplit) != len(dickeys) and len(linesplit) != len(dickeys) + 1:
                continue
            dlist = [float(linesplit[i]) for i in range(len(dickeys))]
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


def computekftnumba(root, Np, L, posox, nk, ntry, natpermol):
    start0 = time.time()
    enk = []
    chk = []
    n1k = []
    n2k = []
    ifprint = False
    G, Gmol, Gmod, Gmodmol = Ggenerateall(nk, Np, L, natpermol)
    with open(root + 'output.out', 'a') as g:
        print('start the computation of the fourier transform of the densities')
        g.write('start the computation of the fourier transform of the densities\n')
    with h5py.File(root + 'dump.h5', 'r') as dump:
        print('tempo di apertira', time.time() - start0)
        snap = list(dump.keys())

        if os.path.exists(root + 'chk.pkl'):
            with open(root + 'enk.pkl', 'rb') as g:
                enk = pk.load(g)
            with open(root + 'chk.pkl', 'rb') as g:
                chk = pk.load(g)
            with open(root + 'n1k.pkl', 'rb') as g:
                n1k = pk.load(g)
            with open(root + 'n2k.pkl', 'rb') as g:
                n2k = pk.load(g)

            lenght = int(len(chk))

            if len(snap) != lenght:
                pass

            print('THE LOADING WAS STOPPED AT THE SNAPSHOT: ', lenght)

        else:
            lenght = 0

        for i in range(lenght + 1, len(snap) + 1):

            start1 = time.time()

            if ifprint:
                print(len(chk))

            datisnap = dump[str(i)][()]

            list1 = np.where(datisnap.T[1] == 1)

            n1 = np.zeros(Np)

            n1[list1] = 1

            n1 -= len(list1[0]) / Np

            list2 = np.where(datisnap.T[1] == 2)

            n2 = np.zeros(Np)

            n2[list2] = 1

            n2 -= len(list2[0]) / Np

            warnings.warn('sto assumendo due specie solamente')

            start2 = time.time()

            if ifprint:
                print('tempo ricerca nel dizionario', start2 - start1)

            poschO, pos = compute.computeposmol(Np, datisnap.transpose(), posox, natpermol)

            dip_mol, cdmol = compute.computemol(Np, datisnap.transpose(), poschO, pos)

            ch_at, pos_at = compute.computeat(Np, datisnap.transpose(), poschO, pos)

            en_at, posatomic, em, endip = compute.computeaten(Np, datisnap.transpose(), pos)

            emp = em / Np * np.ones(Np)

            start3 = time.time()

            if ifprint:
                print('tempo calcolo funzioni', start3 - start2)

            enklist, chklist \
                = numbacomputekft((en_at[:] - emp[:]), (ch_at[:]), posatomic[:, :], pos_at[:, :], L, G, nk)

            n1klist, n2klist \
                = numbacomputekft(n1, n2, pos_at[:, :], pos_at[:, :], L, G, nk)

            # enklist = nfft_adjoint(posatomic[:,0]/L, (en_at[:]- emp[:]), Np)

            enk.append(enklist)

            chk.append(chklist)

            n1k.append(n1klist)

            n2k.append(n2klist)

            start4 = time.time()

            if ifprint:
                print('tempo calcolo ftk', start4 - start3)

            if int(len(chk) - 2) % int(len(snap) / 10) == 0:
                print('got ' + str(len(chk)) + ' snapshot' + '({}%)'.format(int(len(chk) + 1) * 100 // len(snap) + 1))
                print('average elapsed time per snapshot ', (time.time() - start0) / (1 + len(chk)-lenght))
                with open(root + 'output.out', 'a') as z:
                    z.write('got ' + str(len(chk)) + ' snapshot' + '({}%)\n'.format(
                        int(len(chk) + 1) * 100 // len(snap) + 1))
                    z.write('average elapsed time per snapshot {}\n'.format((time.time() - start0) / (1 + len(chk)-lenght)))
                    z.write('tempo ricerca nel dizionario {}\n'.format(start2 - start1))
                    z.write('tempo calcolo funzioni {}\n'.format(start3 - start2))
                    z.write('tempo calcolo ftk {}\n'.format(start4 - start3))

            if int(len(chk) + 1) % int(len(snap) / 4 + 1) == 0:
                with open(root + 'output.out', 'a') as z:
                    with open(root + 'enk.pkl', 'wb+') as g:
                        pk.dump(enk, g)
                    with open(root + 'chk.pkl', 'wb+') as g:
                        pk.dump(chk, g)
                    with open(root + 'n1k.pkl', 'wb+') as g:
                        pk.dump(n1k, g)
                    with open(root + 'n2k.pkl', 'wb+') as g:
                        pk.dump(n2k, g)

                    print('got ' + str(len(chk)) + ' snapshot')
                    print('average elapsed time per snapshot', (time.time() - start0) / (1 + len(chk)-lenght))
                    z.write('got ' + str(len(chk)) + ' snapshot\n')
                    z.write('average elapsed time per snapshot ' + '{}\n'.format(
                        (time.time() - start0) / (1 + len(chk))))

            if len(chk) + 3 == ntry:
                with open(root + 'enk.pkl', 'wb+') as g:
                    pk.dump(enk, g)
                with open(root + 'chk.pkl', 'wb+') as g:
                    pk.dump(chk, g)
                with open(root + 'n1k.pkl', 'wb+') as g:
                    pk.dump(n1k, g)
                with open(root + 'n2k.pkl', 'wb+') as g:
                    pk.dump(n2k, g)
                with open(root + 'output.out', 'a') as g:
                    print('number of total snapshots is', len(chk))
                    print('done')
                    print('elapsed time: ', time.time() - start0)
                    g.write('number of total snapshots is' + '{}\n'.format(len(chk)))
                    g.write('done')
                print('END COMPUTE NTRY')
                return

        with open(root + 'output.out', 'a') as g:
            print('number of total snapshots is', len(chk))
            print('done')
            print('elapsed time: ', time.time() - start0)
            g.write('number of total snapshots is' + '{}\n'.format(len(chk)))
            g.write('done')

        with open(root + 'enk.pkl', 'wb+') as g:
            pk.dump(enk, g)
        with open(root + 'chk.pkl', 'wb+') as g:
            pk.dump(chk, g)
        with open(root + 'n1k.pkl', 'wb+') as g:
            pk.dump(n1k, g)
        with open(root + 'n2k.pkl', 'wb+') as g:
            pk.dump(n2k, g)
        print('END COMPUTE GOOD')
        return


def Ggenerateall(nk, Np, L, natpermol):
    G = np.zeros((nk, 3))
    conta = 0
    G[0] = np.array([0, 0, 0]) / L + 2.335581758729501e-06 / 2 / np.pi / np.sqrt(3.0)
    nkp = int(np.power(nk, 1/3))+1
    for i in range(0, nkp):
        for j in range(0, nkp):
            for k in range(0, nkp):
                if i == 0 and j == 0 and k == 0: continue
                conta += 1
                if conta == nk:
                    Gmod = np.linalg.norm(G, axis=1)
                    return G[:, np.newaxis, :] * np.ones((nk, Np, 3)),\
                           G[:, np.newaxis, :] * np.ones((nk, int(Np / natpermol), 3)),\
                           Gmod[:, np.newaxis] * np.ones((nk, Np)), Gmod[:, np.newaxis] * np.ones((nk, int(Np / natpermol)))
                G[conta] = np.array([i, j, k]) / L + 2.335581758729501e-06 / 2 / np.pi / np.sqrt(3.0)

    Gmod = np.linalg.norm(G, axis=1)
    return G[:, np.newaxis, :] * np.ones((nk, Np, 3)), G[:, np.newaxis, :] * np.ones((nk, int(Np / natpermol), 3)),\
           Gmod[:, np.newaxis] * np.ones((nk, Np)), Gmod[:, np.newaxis] * np.ones((nk, int(Np / natpermol)))


@njit(fastmath=True, parallel=False)
def numbacomputekft(f1, f2, x1, x2, L, G, nk):
    fk1 = [np.sum(f1 * np.exp(1j * 2 * np.sum(x1 * -G[i], axis=1) * np.pi)) for i in range(nk)]
    fk2 = [np.sum(f2 * np.exp(1j * 2 * np.sum(x2 * -G[i], axis=1) * np.pi)) for i in range(nk)]
    return fk1, fk2


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
