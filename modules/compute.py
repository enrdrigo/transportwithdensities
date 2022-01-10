import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os
import time
from numba import njit, jit

# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE POSITION OF THE OXY AND OF THE TWO HYDROGENS AT GIVEN SNAPSHOT. IT ALSO GETS THE POSITION OF THE
# FOURTH PARTICLE IN THE TIP4P/2005 MODEL OF WATER WHERE THERE IS THE CHARGE OF THE OXY (SEE TIP4P/2005 MODEL OF WATER).

@njit(fastmath=True)
def computeposmol(Np, data_array, posox, natpermol):
    nmol = int(Np / natpermol)
    datamol = np.zeros((8, nmol, natpermol))
    datamol = data_array.reshape((8, nmol, natpermol))

    #
    pos = np.zeros((natpermol, 3, nmol))
    posO = np.zeros((3, nmol))
    posH1 = np.zeros((3, nmol))
    posH2 = np.zeros((3, nmol))
    for i in range(natpermol):
        pos[i][0] = np.transpose(datamol[2])[i]
        pos[i][1] = np.transpose(datamol[3])[i]
        pos[i][2] = np.transpose(datamol[4])[i]
        #posH1[0] = np.transpose(datamol[2])[1]
        #posH1[1] = np.transpose(datamol[3])[1]
        #posH1[2] = np.transpose(datamol[4])[1]
        #posH2[0] = np.transpose(datamol[2])[2]
        #posH2[1] = np.transpose(datamol[3])[2]
        #posH2[2] = np.transpose(datamol[4])[2]

    #posO=pos[0]
    #posH1=pos[1]
    #posH2=pos[2]
    #


    if natpermol != 1:
        #
        bisdir = np.zeros((3, nmol))
        bisdir = 2 * pos[0] - pos[1] - pos[2]

        #

        #
        poschO = np.zeros((3, nmol))
        poschO = pos[0] - posox * bisdir / np.sqrt(bisdir[0] ** 2 + bisdir[1] ** 2 + bisdir[2] ** 2)

        #
    else:
        poschO = np.zeros((3, nmol))
        poschO = pos[0]


    return poschO, pos


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE MOLECULAR DIPOLES AND THE CENTER OF MASS OF THE MOLECULES AT GIVEN SNAPSHOT. COMPITING THE MOLECULAR
# DIPOLE WE MUST REMEMBER THAT THE OXY CHARGE IS NOT LOCATED IN THE OXY POSITION (SEE TIP4P/2005 MODEL OF WATER).

@njit(fastmath=True)
def computemol(Np, data_array, poschO, pos):
    natpermol = np.shape(pos)[0]
    nmol = int(Np / natpermol)
    datamol = np.zeros((8, nmol, natpermol))
    datamol = data_array.reshape((8, nmol, natpermol))
    #

    #
    chO = np.zeros(nmol)
    chH1 = np.zeros(nmol)
    chH2 = np.zeros(nmol)
    ch = np.zeros((natpermol, nmol))
    for i  in range(natpermol):
        ch[i] = np.transpose(datamol[5])[i]


    #chH1 = ch[1]  # np.transpose(data_array[5])[1]
    #chH2 = ch[2]  # np.transpose(data_array[5])[2]
    #

    #
    mass = np.array([15.9994, 1.008, 1.008])
    cdmmoln = np.zeros((natpermol, 3, nmol))
    for i in range(natpermol):
        cdmmoln[i] = pos[i]*mass[i]/np.sum(mass)
    cdmmol = np.zeros((3, nmol))
    cdmmol = np.sum(cdmmoln, axis=0)
    #

    #
    pos_mch = np.zeros((3, nmol))
    pos_mch0 = poschO * ch[0]

    pos_mchn = np.zeros((natpermol, 3, nmol))
    for i in range(natpermol-1):
        pos_mchn[i] = pos[i+1]*ch[i+1]
    pos_mch = pos_mch0 + np.sum(pos_mchn, axis=0)
    #

    #
    dip_mol0 = np.zeros((3, nmol))
    dip_mol0 = pos_mch
    #

    return np.transpose(dip_mol0), np.transpose(cdmmol)


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE CHARGE AND ATOMIC POSITION ARRAYS OF THE ATOMS AT A GIVEN SNAPSHOT. THE OXY POSITION IS SHIFTED ACCORDING
# TO THE TIP4P/2005 MODEL OF WATER.


def computeat(Np, data_array, poschO, pos):
    natpermol = np.shape(pos)[0]
    nmol = int(Np / natpermol)

    #
    chat = np.zeros(Np)
    chat = data_array[5]
    #

    #
    posm = np.zeros((natpermol, nmol, 3))
    posm[0] = np.transpose(poschO)
    for i in range(1, natpermol):
        posm[i] = np.transpose(pos[i])

    #

    #
    pos_at = np.zeros((3, Np))
    test = np.transpose(posm)
    pos_at = test.reshape((3, Np))
    #

    # THIS IS IN FACT THE CHARGE TIMES A PHASE WHERE GAT = 2 * np.pi * np.array((1e-8, 1e-8, 1e-8)) / L. I DO THIS
    # IN ORDER TO COMPUTE PROPERLY THE STATIC DIELECTRIC CONSTANT VIA THE FOURIER TRANFORM OR THE CHARGE OVER
    # THE MODULUS OF G,  AT G \APPROX 0
    ch_at = np.zeros(Np)
    ch_at = chat
    #

    return ch_at, np.transpose(pos_at)

@njit(fastmath=True)
def computeaten(Np, data_array, pos):
    natpermol = np.shape(pos)[0]
    nmol = int(Np / natpermol)
    #
    datamol = np.zeros((8, nmol, natpermol))
    datamol = data_array.reshape((8, nmol, natpermol))
    #
    en0 = np.zeros(nmol)
    enH1 = np.zeros(nmol)
    enH2 = np.zeros(nmol)
    en = np.zeros((3, nmol))
    for i in range(natpermol):
        en[i] = np.transpose(datamol[6])[i] + np.transpose(datamol[7])[i]
        #enH1 = np.transpose(datamol[6])[1] + np.transpose(datamol[7])[1]
        #enH2 = np.transpose(datamol[6])[2] + np.transpose(datamol[7])[2]

    #
    enat = np.zeros(Np)
    enat = data_array[6] + data_array[7]
    #

    #
    pos_at = np.zeros((3, Np))
    pos_at[0] = data_array[2]
    pos_at[1] = data_array[3]
    pos_at[2] = data_array[4]
    #

    #
    en_at = np.zeros(Np)
    en_at = enat
    #

    endip = np.zeros((3, nmol))
    endipn = np.zeros((natpermol, 3, nmol))
    for i in range(natpermol):
        endipn[i] = pos[i]*(en[i]-np.sum(enat)/Np)#poschO*(enO-np.sum(enat)/Np) + posH1*(enH1-np.sum(enat)/Np) + posH2*(enH2-np.sum(enat)/Np)
    endip = np.sum(endipn, axis=0)

    return en_at, np.transpose(pos_at), np.sum(enat), np.transpose(endip)



def computekft(root, filename, Np, L, posox, nk, ntry, natpermol):
    start = time.time()
    enk = []
    dipenkx = []
    dipenky = []
    chk = []
    dipkx = []
    dipky = []
    with open(root + 'output.out', 'a') as g:
        print('start the computation of the fourier transform of the densities')
        g.write('start the computation of the fourier transform of the densities\n')
    with open(root+filename, 'r') as f:
        line = f.readline()

        if os.path.exists(root + 'chk.pkl'):
            with open(root + 'enk.pkl', 'rb') as g:
                enk = pk.load(g)
            with open(root + 'dipenkx.pkl', 'rb') as g:
                dipenkx = pk.load(g)
            with open(root + 'dipenky.pkl', 'rb') as g:
                dipenky = pk.load(g)
            with open(root + 'chk.pkl', 'rb') as g:
                chk = pk.load(g)
            with open(root + 'dipkx.pkl', 'rb') as g:
                dipkx = pk.load(g)
            with open(root + 'dipky.pkl', 'rb') as g:
                dipky = pk.load(g)

            lenght = int(len(enk) / 3)

            print('THE LOADING WAS STOPPED AT THE SNAPSHOT: ', lenght)
            for p in range((Np + 9) * lenght):
                f.readline()

        while line != '':

            d = []
            for p in range(Np+9):
                if len(line.split(' ')) != 8 and len(line.split(' ')) != 9:
                    line = f.readline()
                    continue
                dlist = [float(line.split(' ')[i]) for i in range(8)]
                line = f.readline()
                d.append(dlist)

            if len(d) == 0:
                with open(root + 'enk.pkl', 'wb+') as g:
                    pk.dump(enk, g)
                with open(root + 'dipenkx.pkl', 'wb+') as g:
                    pk.dump(dipenkx, g)
                with open(root + 'dipenky.pkl', 'wb+') as g:
                    pk.dump(dipenky, g)
                with open(root + 'chk.pkl', 'wb+') as g:
                    pk.dump(chk, g)
                with open(root + 'dipkx.pkl', 'wb+') as g:
                    pk.dump(dipkx, g)
                with open(root + 'dipky.pkl', 'wb+') as g:
                    pk.dump(dipky, g)
                print('END READ FILE')
                print('got ' + str(len(chk) / 3) + ' snapshot')
                print('elapsed time: ', time.time() - start)
                with open(root + 'output.out', 'a') as z:
                    z.write('got ' + str(len(chk) / 3) + ' snapshot\n')

                return # len(chk), np.transpose(np.array(enk)), np.transpose(np.array(dipenkx)), np.transpose(np.array(dipenky)), np.transpose(np.array(chk)), np.transpose(np.array(dipkx)), np.transpose(np.array(dipky))

            elif len(d) != Np:
                with open(root + 'enk.pkl', 'wb+') as g:
                    pk.dump(enk, g)
                with open(root + 'dipenkx.pkl', 'wb+') as g:
                    pk.dump(dipenkx, g)
                with open(root + 'dipenky.pkl', 'wb+') as g:
                    pk.dump(dipenky, g)
                with open(root + 'chk.pkl', 'wb+') as g:
                    pk.dump(chk, g)
                with open(root + 'dipkx.pkl', 'wb+') as g:
                    pk.dump(dipkx, g)
                with open(root + 'dipky.pkl', 'wb+') as g:
                    pk.dump(dipky, g)

                print(len(d))
                print('STOP: THE SNAPSHOT '+str(int(len(enk)/3)+1)+' DOES NOT HAVE ALL THE PARTICLES')
                print('got ' + str(len(chk) / 3) + ' snapshot')
                with open(root + 'output.out', 'a') as z:
                    z.write('got ' + str(len(chk) / 3) + ' snapshot\n')
                return #len(chk), np.transpose(np.array(enk)), np.transpose(np.array(dipenkx)), np.transpose(np.array(dipenky)), np.transpose(np.array(chk)), np.transpose(np.array(dipkx)), np.transpose(np.array(dipky))

            datisnap = np.array(d)

            if len(chk) == ntry * 3:
                with open(root + 'enk.pkl', 'wb+') as g:
                    pk.dump(enk, g)
                with open(root + 'dipenkx.pkl', 'wb+') as g:
                    pk.dump(dipenkx, g)
                with open(root + 'dipenky.pkl', 'wb+') as g:
                    pk.dump(dipenky, g)
                with open(root + 'chk.pkl', 'wb+') as g:
                    pk.dump(chk, g)
                with open(root + 'dipkx.pkl', 'wb+') as g:
                    pk.dump(dipkx, g)
                with open(root + 'dipky.pkl', 'wb+') as g:
                    pk.dump(dipky, g)
                with open(root + 'output.out', 'a') as g:
                    print('number of total snapshots is', len(chk) / 3)
                    print('done')
                    g.write('number of total snapshots is' + '{}\n'.format(len(chk) / 3))
                    g.write('done')
                print('END READ. NO MORE DATA TO LOAD. SEE NTRY')
                return #len(chk), np.transpose(np.array(enk)), np.transpose(np.array(dipenkx)), np.transpose(np.array(dipenky)), np.transpose(np.array(chk)), np.transpose(np.array(dipkx)), np.transpose(np.array(dipky))

            poschO, pos = computeposmol(Np, datisnap.transpose(), posox, natpermol)

            dip_mol, cdmol = computemol(Np, datisnap.transpose(), poschO, pos)

            ch_at, pos_at = computeat(Np, datisnap.transpose(), poschO, pos)

            en_at, posatomic, em, endip = computeaten(Np, datisnap.transpose(), pos)

            emp = em/Np*np.ones(Np)

            enklist = [np.sum((en_at[:]- emp) * np.exp(1j * posatomic[:, 0] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            enk.append(enklist)

            enklist = [np.sum((en_at[:] - emp) * np.exp(1j * posatomic[:, 1] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            enk.append(enklist)

            enklist = [np.sum((en_at[:] - emp) * np.exp(1j * posatomic[:, 2] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            enk.append(enklist)

            dipenkxlist = [np.sum((endip[:, 0]) * np.exp(1j * cdmol[:, 0] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipenkx.append(dipenkxlist)

            dipenkxlist = [np.sum((endip[:, 1]) * np.exp(1j * cdmol[:, 1] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipenkx.append(dipenkxlist)

            dipenkxlist = [np.sum((endip[:, 2]) * np.exp(1j * cdmol[:, 2] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipenkx.append(dipenkxlist)

            dipenkylist = [np.sum((endip[:, 1]) * np.exp(1j * cdmol[:, 0] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipenky.append(dipenkylist)

            dipenkylist = [np.sum((endip[:, 2]) * np.exp(1j * cdmol[:, 0] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipenky.append(dipenkylist)

            dipenkylist = [np.sum((endip[:, 2]) * np.exp(1j * cdmol[:, 1] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipenky.append(dipenkylist)

            chklist = [np.sum((ch_at[:]) * np.exp(1j * pos_at[:, 0] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            chk.append(chklist)

            chklist = [np.sum((ch_at[:]) * np.exp(1j * pos_at[:, 1] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            chk.append(chklist)

            chklist = [np.sum((ch_at[:]) * np.exp(1j * pos_at[:, 2] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            chk.append(chklist)

            dipkxlist = [np.sum((dip_mol[:, 0]) * np.exp(1j * cdmol[:, 0] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipkx.append(dipkxlist)

            dipkxlist = [np.sum((dip_mol[:, 1]) * np.exp(1j * cdmol[:, 1] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipkx.append(dipkxlist)

            dipkxlist = [np.sum((dip_mol[:, 2]) * np.exp(1j * cdmol[:, 2] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipkx.append(dipkxlist)

            dipkylist = [np.sum((dip_mol[:, 1]) * np.exp(1j * cdmol[:, 0] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipky.append(dipkylist)

            dipkylist = [np.sum((dip_mol[:, 2]) * np.exp(1j * cdmol[:, 0] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipky.append(dipkylist)

            dipkylist = [np.sum((dip_mol[:, 2]) * np.exp(1j * cdmol[:, 1] * 2 * -i * np.pi / L), axis=0) for i in range(nk)]

            dipky.append(dipkylist)

            with open(root + 'output.out', 'a') as z:
                if int(len(chk)/3) % 5000 == 0:

                    with open(root + 'enk.pkl', 'wb+') as g:
                        pk.dump(enk, g)
                    with open(root + 'dipenkx.pkl', 'wb+') as g:
                        pk.dump(dipenkx, g)
                    with open(root + 'dipenky.pkl', 'wb+') as g:
                        pk.dump(dipenky, g)
                    with open(root + 'chk.pkl', 'wb+') as g:
                        pk.dump(chk, g)
                    with open(root + 'dipkx.pkl', 'wb+') as g:
                        pk.dump(dipkx, g)
                    with open(root + 'dipky.pkl', 'wb+') as g:
                        pk.dump(dipky, g)

                    print('got '+str(len(chk)/3)+' snapshot')
                    print('average elapsed time per snapshot', (time.time()-start)/(len(chk)/3))
                    z.write('got '+str(len(chk)/3)+' snapshot\n')
                    z.write('average elapsed time per snapshot'+'{}\n'.format((time.time()-start)/(len(chk)/3)))

            if len(chk) == ntry*3:
                with open(root + 'enk.pkl', 'wb+') as g:
                    pk.dump(enk, g)
                with open(root + 'dipenkx.pkl', 'wb+') as g:
                    pk.dump(dipenkx, g)
                with open(root + 'dipenky.pkl', 'wb+') as g:
                    pk.dump(dipenky, g)
                with open(root + 'chk.pkl', 'wb+') as g:
                    pk.dump(chk, g)
                with open(root + 'dipkx.pkl', 'wb+') as g:
                    pk.dump(dipkx, g)
                with open(root + 'dipky.pkl', 'wb+') as g:
                    pk.dump(dipky, g)
                with open(root + 'output.out', 'a') as g:
                    print('number of total snapshots is', len(chk) / 3)
                    print('done')
                    print('elapsed time: ', time.time() - start)
                    g.write('number of total snapshots is' + '{}\n'.format(len(chk) / 3))
                    g.write('done')
                print('END READ NTRY')
                return #len(chk), np.transpose(np.array(enk)), np.transpose(np.array(dipenkx)), np.transpose(np.array(dipenky)), np.transpose(np.array(chk)), np.transpose(np.array(dipkx)), np.transpose(np.array(dipky))

        with open(root + 'output.out', 'a') as g:
            print('number of total snapshots is', len(chk)/3)
            print('done')
            print('elapsed time: ', time.time() - start)
            g.write('number of total snapshots is'+'{}\n'.format(len(chk)/3))
            g.write('done')

        with open(root+'enk.pkl', 'wb+') as g:
            pk.dump(enk, g)
        with open(root+'dipenkx.pkl', 'wb+') as g:
            pk.dump(dipenkx, g)
        with open(root+'dipenky.pkl', 'wb+') as g:
            pk.dump(dipenky, g)
        with open(root+'chk.pkl', 'wb+') as g:
            pk.dump(chk, g)
        with open(root+'dipkx.pkl', 'wb+') as g:
            pk.dump(dipkx, g)
        with open(root+'dipky.pkl', 'wb+') as g:
            pk.dump(dipky, g)
        print('END READ FILE GOOD')
        return  # len(chk)/3, np.transpose(np.array(enk)), np.transpose(np.array(dipenkx)), np.transpose(np.array(dipenky)), np.transpose(np.array(chk)), np.transpose(np.array(dipkx)), np.transpose(np.array(dipky))
