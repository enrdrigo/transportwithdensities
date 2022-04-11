import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# READS THE DATA FROM THE LAMMPS OUTPUT AND SAVE IT IN A BINARY FORM. IT CAN TAKE VERY LONG.

def getdatafromfile(filename, root, Np):
    with  open(root + filename, 'r') as f:
        line = f.readline()
        d = []
        g = open('file.out', 'a')
        g.write('start translation in bin\n')
        g.close()
        while (line != ''):
            if len(line.split(' ')) != 8:
                line = f.readline()
                continue
            dlist = [float(x.strip('\n')) for x in line.split(' ')]
            line = f.readline()
            d.append(dlist)
        g = open('file.out', 'a')
        g.write('done translation in bin\n')
        g.close()

    return d


# ----------------------------------------------------------------------------------------------------------------------
# GETS FROM THE LAMMPS OUTPUT THE NUMBER OF ATOMS IN THE SIMULATION

def getNpart(filename, root):
    with open(root + filename, 'r') as f:
        oldline = 'noline'
        for i in range(15):
            line = f.readline()
            if oldline == 'ITEM: NUMBER OF ATOMS\n':
                Npart = int(line.split()[0])
                f.close()
                return Npart
            oldline = line


# ----------------------------------------------------------------------------------------------------------------------
# GETS FROM THE LAMMPS OUTPUT THE DIMENTION OF THE SIDE OF THE SIMULATION BOX

def getBoxboundary(filename, root):
    with open(root + filename, 'r') as f:
        oldline = 'noline'
        for i in range(15):
            line = f.readline()
            if oldline == 'ITEM: BOX BOUNDS pp pp pp\n':
                (Linfx, Lmaxx) = (float(line.split()[0]), float(line.split()[1]))
                f.close()
                (Linfy, Lmaxy) = (float(line.split()[0]), float(line.split()[1]))
                f.close()
                (Linfz, Lmaxz) = (float(line.split()[0]), float(line.split()[1]))
                f.close()
                return np.array([Lmaxx - Linfx, Lmaxy - Linfy, Lmaxz - Linfz]), np.array([Linfx, Linfy, Linfz])
            oldline = line


# ----------------------------------------------------------------------------------------------------------------------
# GETS THE NUMBER OF SNAPSHOT THAT WE ARE CONSIDERING AND PERFORMS A RESHAPE OF THE DATA ARRAY SO THAT WE HAVE FOR EACH
# SNAPSHOT A MATRIX WITH THE POSITION AND THE CHARGES OF THE MOLECULES

def getNsnap(dati, Np):
    nsnap = int(len(dati) / Np)
    print('number of snapshot in the trajectory: ', nsnap)
    return nsnap


def getinitialize(filename, root, posox, nk, ntry):
    with open(root+'output.out', 'w+') as f:
        Npart = getNpart(filename, root)
        L, Linf = getBoxboundary(filename, root)
        print('root: ', root)
        print('filename: ', filename)
        print('number of particles: ', Npart)
        print('size of the box in angstrom: ', L)
        print('relative distance of the oxy charge: ', posox)
        print('number of k points: ', nk)
        print('number of snapshot (if -1 gets all of them): ', ntry)
        f.write('root: '+'{}\n'.format(root))
        f.write('filename: '+'{}\n'.format(filename))
        f.write('number of particles: '+'{}\n'.format(Npart))
        f.write('size of the box in angstrom: '+'{}\n'.format(L))
        f.write('relative distance of the oxy charge: '+'{}\n'.format(posox))
        f.write('number of k points: '+'{}\n'.format(nk))
        f.write('number of snapshot (if -1 gets all of them): '+'{}\n'.format(ntry))
        inp = {'root': root, 'filename': filename, 'N': Npart, 'size': L, 'position of the ox': posox, 'number of k': nk, 'number of snapshots': ntry}
        return inp