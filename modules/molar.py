import numpy as np
import time
import h5py
import os
from modules import initialize
import warnings




def read_log_lammps(root, filename):
    print('Start read_log_lammps routine')
    start = time.time()
    print(
        'Starting the reading for the file '+filename+' with tehrnodynamic quatities like total energy, temperature, pressure or enthalpy. The ouput file is a dictionary saved in '
        + root + filename +'.npy')
    datadic = {}
    with open(root + filename, 'r') as f:

        startcollect = False

        for index, line in enumerate(f):

            linesplit = []

            for i in line.split(' '):

                if i != '': linesplit.append(i)

            if linesplit[0] == 'Step':
                dickeys = []
                for i in linesplit[:]:
                    if str(i) == '\n': continue
                    dickeys.append(str(i))
                data = [[] for i in dickeys]
                print(dickeys)

            if linesplit[0] == 'Step': startcollect = True; continue

            if linesplit[0] == 'Loop': break

            if startcollect:
                for i in range(len(dickeys)):
                    if linesplit[i]=='\n': continue
                    data[i].append(float(linesplit[i]))

        datadic = {dickeys[0]: np.array(data[0])}

        for i in dickeys[1:]:
            datadic[i] = np.array(data[dickeys.index(i)])
    print('elapsed time: ', time.time() - start)
    np.save(root + filename + '.npy', datadic)
    print('End read_log_lammps routine')
    return datadic


def molar(root, filename, Np, nblocks):
    start = time.time()

    with open(root + 'molaroutput.out', 'w') as g:
        g.write('\nStart molar routine')
        g.write('\nThe routine reads the dump.h5 file with the unscaled atomic positions and energies per atom.\n ' +
          'Computes the partial volumes (in units of the volume per particle) and the partial energies as described by Pablo Debenedetti. ')
    print('Start molar routine')
    print('The routine reads the dump.h5 file with the unscaled atomic positions and energies per atom.\n ' +
          'Computes the partial volumes (in units of the volume per particle) and the partial energies as described by Pablo Debenedetti.')
    if os.path.exists(root + 'dump.h5'):
        pass
    else:
        raise ValueError('dump.h5 file not available! create it with the routine read_dump')

    L, L_min = initialize.getBoxboundary(filename,
                                         root)

    with h5py.File(root + 'dump.h5', 'r') as dump:

        fetta = {'x': 0, 'y': 1, 'z': 2}
        portions = ['x', 'y', 'z']
        startr = time.time()
        snap = [[] for i in range(dump['data'].len())]
        startr = time.time()
        energies = np.zeros((len(snap), len(portions)))
        enmean = np.zeros(len(snap))
        N1 = np.zeros((len(snap), len(portions)))
        N2 = np.zeros((len(snap), len(portions)))
        enm1 = np.zeros((len(snap), len(portions)))
        enm2 = np.zeros((len(snap), len(portions)))
        vpp1 = np.zeros((len(snap), len(portions)))
        vpp2 = np.zeros((len(snap), len(portions)))

        for i in range(1, len(snap) + 1):

            if i % int(len(snap) / 10) == 0:
                with open(root + 'molaroutput.out', 'a') as g:
                    g.write('\n'+str((i * 100) // len(snap)) + '% done in {}'.format(time.time()-startr))
                print((i * 100) // len(snap), '% done in {}'.format(time.time()-startr))
                startr=time.time()

            j = i - 1
            dumpdata = dump['data'][j].T
            enmean[j] = dumpdata[6].sum() + dumpdata[7].sum()
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
                enm1[j, sin] =(dumpdata[6][list_fetta_sp1].sum() + dumpdata[7][list_fetta_sp1].sum())/len(list_fetta_sp1)
                enm2[j, sin] =(dumpdata[6][list_fetta_sp2].sum() + dumpdata[7][list_fetta_sp2].sum())/ len(list_fetta_sp2)
                vpp1[j, sin] = len(list_fetta_sp1)/len(list_fetta)
                vpp2[j, sin] = len(list_fetta_sp2)/len(list_fetta)


    vb = np.zeros((nblocks, len(portions), 2))
    ub = np.zeros((nblocks, len(portions), 2))
    xb = np.zeros((nblocks, len(portions), 2))

    for b in range(nblocks):

        energiesb = np.zeros((int(len(snap) / nblocks), 3))
        N1b = np.zeros((int(len(snap) / nblocks), 3))
        N2b = np.zeros((int(len(snap) / nblocks), 3))

        energiesb = energies[b * int(len(snap) / nblocks):(b + 1) * int(len(snap) / nblocks), :]
        N1b = N1[b * int(len(snap) / nblocks):(b + 1) * int(len(snap) / nblocks)]
        N2b = N2[b * int(len(snap) / nblocks):(b + 1) * int(len(snap) / nblocks)]

        n1 = N1b.mean(axis=0)
        n2 = N2b.mean(axis=0)
        energy = energiesb.mean(axis=0)

        #i, l inidici di specie, m il tempo, n la fetta. Medio su axis=2, quindi sul tempo. delta ha le dimensioni di #specie, #specie, #fetta

        delta = np.einsum('imn,lmn->ilmn',
                          np.array([N1b - n1, N2b - n2]),
                          np.array([N1b - n1, N2b - n2])).mean(axis=2)

        #x ha le dimensioni #fetta, #specie
        x = np.array([n1 / (n1 + n2), n2 / (n1 + n2)]).transpose((1, 0))

        xb[b] = x

        #l indice di tempo, m di fetta, i indice di specie. alpha ha le dimensioni di #fetta, #specie
        alpha = np.einsum('lm,ilm->mli',
                          energiesb - energy,
                          np.array([N1b - n1, N2b - n2])).mean(axis=1)

        #print(np.mean((energiesb-energy)*np.array([N1b-n1, N2b-n2]), axis=1))

        #la funzione di inversione di matrici permette di fare l'inversione vettoriale di una lista di matrici, l'indice della matrice sara` il primo
        #ex. (N, M, M). dinv ha le dimensioni, #fetta, #specie, #specie
        try:
            dinv = np.linalg.inv(delta.transpose((2, 0, 1)))
        except:
            print('probabilmente quello che e` successo e` che nel blocco in una fetta non e` mai cambiato il numero' +
                  ' di atomi  per una specie. puoi provare a dimunuire il numero di blocchi, cosi` e` piu` probabile' +
                  ' che almeno in un istante ci sia una fluttuazione. se non migliora allunga la traiettoria o aumenta'+
                  ' il numero di atomi.')
            print('block number', b)
            print('number of particle per species', N1b, N2b)
            raise ValueError('Singular matrix!')


        #i inisice di fetta, j indice di specie, k indice di specie.  den ha le dimensioni di #fetta
        den = np.einsum('ij,ik,ijk->i',
                        x,
                        x,
                        dinv) ** -1

        # partial molar volumes, in unita' di volume per particella
        #i indice di fetta, l indice di specie. v ha le dimensioni di #fetta, #specie
        v = np.einsum('il,i->il',
                      np.einsum('ilm,il->im',# i indice di fetta l indice di specie, m indice di specie, sommo sulle specie
                                dinv,
                                x),
                      den)

        #vb ha le dimensioni di #blocchi, #fetta, #specie
        vb[b] = v

        # partial molar energies
        #i indice di fetta, l indice di specie. u ha le dimensioni di #fetta, #specie
        u = np.einsum('i,il->il', energy / (n1 + n2), v) + \
            np.einsum('ab,abc->ac',#a indice di fetta, b indice di specie, c indice di specie
                      alpha,
                      dinv - np.einsum('abc,a->abc',#a indice di fetta. b indice di specie, c indice di specie
                                       np.einsum('ab,abc,ad,adf->acf',#a indice di fetta, b indice di specie, c indice di specie, d indice di specie, f indice di specie
                                                 x,
                                                 dinv,
                                                 x,
                                                 dinv),
                                       den))

        #ub ha le dimensioni #blocchi, #fetta, #specie
        ub[b] = u
    with open(root + 'molaroutput.out', 'a') as g:
        g.write('\npartial volumes' +
          str(vb.mean(axis=1).mean(axis=0)) +
          ' \n euler relation for the partial volumes:' +
          str(np.sum(vb.mean(axis=1) * xb.mean(axis=1), axis=1).mean(axis=0)))
        g.write('\npartial energies' +
          str(ub.mean(axis=1).mean(axis=0)) +
          ' \n euler relations for the partial energies:\n' +
          str(enmean.mean() / Np) +
          str(np.sum(ub.mean(axis=1) * xb.mean(axis=1), axis=1).mean(axis=0)))
        g.write('\nmean energies per species:' + str(enm1.mean()) + str(enm2.mean()))
        g.write('\nelapsed time: ' + str(time.time() - start))
        g.write('\nEnd molar routine')
    print('partial volumes',
          vb.mean(axis=1).mean(axis=0),
          ', \n euler relation for the partial volumes:',
          np.sum(vb.mean(axis=1) * xb.mean(axis=1), axis=1).mean(axis=0))
    print('partial energies',
          ub.mean(axis=1).mean(axis=0),
          ', \n euler relations for the partial energies:',
          enmean.mean() / Np,
          np.sum(ub.mean(axis=1) * xb.mean(axis=1), axis=1).mean(axis=0))
    print('mean energies per species:', enm1.mean(), enm2.mean())
    print('elapsed time: ', time.time() - start)
    print('End molar routine')
    return vb, ub, xb


def wrappos(posunw, L, L_min):
    return (np.mod((posunw.T - L_min), L) / L).T


def molar_enthalpy(root, filename, filename_log, volume, Np, nblocks, UNITS='metal'):
    print('Start molar_enthalpy routine')
    start=time.time()
    volumepp = volume / Np

    if os.path.exists(root + 'molar.npy'):
        res = np.load(root + 'molar.npy', allow_pickle=True).item()
        return res['molar enthalpies']
    else: pass
    if os.path.exists(root + filename_log + '.npy'):
        dic_data_log = np.load(root + filename_log + '.npy', allow_pickle='TRUE').item()
    else:
        dic_data_log = read_log_lammps(root=root,
                                    filename=filename_log)



    print(dic_data_log.keys())

    v, u, x = molar(root,
                 filename,
                 Np,
                 nblocks)

    warnings.warn('conversion atm*\AA**3_to_Kcal/mol, lammps real units')
    warnings.warn('conversion bar*\AA**3_to_eV, lammps metal units')

    facreal = 1.0125e5 / 1.0e30 * 6.022e23 / 4186
    facmetal = 1e5 / 1.0e30 / 1.60218e-19
    faclj = 1
    if UNITS == 'metal':
        fac = facmetal
    if UNITS == 'real':
        fac = facreal
    if UNITS == 'lj':
        fac = faclj

    h = u + np.mean(dic_data_log['Press']) * volumepp * v * fac

    eru = u.mean(axis=1).std(axis=0) / u.mean(axis=1).mean(axis=0) / np.sqrt(3 * nblocks)
    print('relative percentage std of the partial eneergies %',
          eru / u.mean(axis=1).mean(axis=0) * 100)

    errvol = v.mean(axis=1).std(axis=0) / v.mean(axis=1).mean(axis=0) / np.sqrt(3 * nblocks)
    print('relative percentage std of the partial volumes %',
          errvol * 100)

    errpress = (dic_data_log['Press']).std() / dic_data_log['Press'].mean() / np.sqrt(len(dic_data_log['Press']))
    print('relative percentage std of the pressure %',
          (dic_data_log['Press']).std() / dic_data_log['Press'].mean() * 100 / np.sqrt(len(dic_data_log['Press'])))

    PV = (np.mean(dic_data_log['Press']) * volumepp * v * fac).mean(axis=1).mean(axis=0)
    print('PV contribution to the partial enthalpies',
          PV)

    print('partial enthalpies ',
          h.mean(axis=1).mean(axis=0),
          ',\n Euler relation for the partial enthalpies',
          h.mean(axis=1).mean(axis=0)[0]*x.mean(axis=1).mean(axis=0)[0] + h.mean(axis=1).mean(axis=0)[1] *x.mean(axis=1).mean(axis=0)[1],
          (np.mean(dic_data_log['TotEng']) + np.mean(dic_data_log['Press']) * volume * fac) / Np)

    print('std partial enthalpies', np.sqrt((eru)**2+(PV*(errvol+errpress))**2))
    print('Elapsed time', time.time() - start)
    print('End molar_enthalpy routine')

    res = {}
    res = {'molar enthalpies': h,
           'molar volumes': v*volumepp,
           'molar energies': u}

    np.save(root + 'molar.npy', res)

    return h
