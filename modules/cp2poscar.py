#!/bin/env python
##### coding=utf-8
import numpy as np
import argparse
import warnings
from subprocess import check_output
import re
import os

BOHR = 0.529177249  # Bohr constant in Angstrom
# TAU  = 4.8378e-5  # tau_PW constant in ps
TAU = 0.5 * 4.8378e-5  # tau_CP constant in ps
HARTREE = 27.211386245988  # eV
eV = 1.60217662  # 10^-19 J
bar = 1.0e-2 / eV  # eV/Ang^3


def read_input_type(prefix, natoms_per_type, species):
    #       print(int(check_output(["wc", "-l", prefix + '.input']).decode("utf8").split()[0]),sum(natoms_per_type))
    #       if (int(check_output(["wc", "-l", prefix + '.input']).decode("utf8").split()[0]) != sum(natoms_per_type)):
    #               raise RuntimeError("Number of atoms in input does not match the one given in input")
    type_array = np.zeros(sum(natoms_per_type), dtype=int)
    typemap = {}
    for index, type in enumerate(species):
        typemap[type] = index
    with open(prefix + '.input', 'r') as filein:
        for i in range(4000):
            dline = filein.readline().split()
            if (len(dline) != 0 and dline[0] == "ATOMIC_POSITIONS"):
                for iatom in range(sum(natoms_per_type)):
                    dline = filein.readline().split()
                    type_array[iatom] = typemap[dline[0]]
                return type_array


def read_file_pos_vel(prefix, natoms, nstep=None):
    """
    Legge i file di output di quantum espresso (cartella dove sono posizionati i vari restart)
    Per esempio, se il prefisso is KCl-512:
    namepos is KCl-512.pos
    namevel is KCl-512.vel
    nstep is il numero di timestep da leggere (NON is determinato automaticamente!)
    natoms is il numero di atomi nella simulazione (NON is determinato automaticamente!)

    ritorna una lista, una per ogni timestep, con il contenuto:
        [timestep,tempo]       [ posizioni ]               [velocita']
    dove [timestep,tempo] is una coppia di numeri, posizioni is un array numpy con le posizioni,
    e velocita' is un array numpy
    """

    def file_length(filename):
        i = -1
        blank = 0
        with open(filename) as f:
            for i, l in enumerate(f, 1):
                if len(l) == 1:
                    blank += 1
                pass
        return i - blank

    if nstep is None:
        nlines = int(check_output(["wc", "-l", prefix + '.pos']).decode("utf8").split()[0])
        nstep = nlines // (natoms + 1)
        # nstep = file_length(prefix + '.evp') - 1
        print("nstep not set: using all the steps in .pos file: nstep = {}".format(nstep))

    # ToDo: possibilita' di leggere i file dal fondo
    #       if nstep < 0:
    #               reverse = True
    #               nstep = -nstep
    #       else:
    #               reverse = False
    #
    data = {}

    filethe = open(prefix + '.evp', 'r')
    data['step'] = np.zeros(nstep, dtype=np.int64)
    data['time'] = np.zeros(nstep, dtype=np.float64)
    data['ekinc'] = np.zeros(nstep, dtype=np.float64)
    data['Tcell'] = np.zeros(nstep, dtype=np.float64)
    data['Tion'] = np.zeros(nstep, dtype=np.float64)
    data['econt'] = np.zeros(nstep, dtype=np.float64)
    data['epot'] = np.zeros(nstep, dtype=np.float64)
    # filethe.readline()  # skip first line

    filepos = open(prefix + '.pos', 'r')
    data['pos'] = np.zeros((nstep, natoms, 3), dtype=np.float64)

    filevel = open(prefix + '.vel', 'r')
    data['vel'] = np.zeros((nstep, natoms, 3), dtype=np.float64)

    try:
        filefor = open(prefix + '.for', 'r')
        read_force = True
        data['for'] = np.zeros((nstep, natoms, 3), dtype=np.float64)
    except IOError as err:
        err = re.sub(r'\[.*\]', '', str(err))
        print('Warning!' + err + '. The .for file is not present: it will be ignored')
        read_force = False

    try:
        filestr = open(prefix + '.str', 'r')
        read_stress = True
        data['str'] = np.zeros((nstep, 9), dtype=np.float64)
    except IOError as err:
        err = re.sub(r'\[.*\]', '', str(err))
        print('Warning!' + err + '. The .str file is not present: it will be ignored')
        read_stress = False

    filecel = open(prefix + '.cel', 'r')
    data['cell'] = np.zeros((nstep, 9), dtype=np.float64)

    istep = 0

    while (istep < nstep):
        linethe = filethe.readline()
        # print(linethe)
        linepos = filepos.readline()
        linevel = filevel.readline()
        if read_force: linefor = filefor.readline()
        if read_stress: linestr = filestr.readline()
        linecel = filecel.readline()
        if (len(linethe) == 0) or (len(linepos) == 0) or (len(linevel) == 0) or (len(linecel) == 0):  # EOF
            if read_force:
                if (len(linefor) == 0):
                    raise RuntimeError("End Of file")
            if read_stress:
                if (len(linestr) == 0): raise RuntimeError("End Of file")

        # controllo per commenti
        if (linethe.split()[0] == '#'):
            print("Comment found in {}.evp at line {}. Please check that this is correct.".format(prefix, istep + 1))
            linethe = filethe.readline()
        # lettura thermo
        values = np.array(linethe.split(), dtype=np.float)
        if len(values):
            # print istep, values[0], len(data['step'])
            data['step'][istep] = int(values[0])
            data['time'][istep] = values[1]
            if istep == 1:
                deltat = data['time'][1] - data['time'][0]
            data['ekinc'][istep] = values[2]
            data['Tcell'][istep] = values[3]
            data['Tion'][istep] = values[4]
            data['epot'][istep] = values[5]
            data['econt'][istep] = values[8]
        else:
            istep -= 1

        # lettura posizioni
        # values = np.array(linepos.split(), dtype = np.float)
        values = linepos.split()
        # print linepos
        # print values, data['step'][istep]
        if len(values):
            if (data['step'][istep] != int(values[0])):
                print(data['step'][istep], int(values[0]))
                raise RuntimeError("Different timesteps between files of positions and thermo")
            for iatom in range(natoms):
                linepos = filepos.readline()
                values = np.array(linepos.split())
                data['pos'][istep, iatom, :] = values[:]

        # lettura velocity
        # values = np.array(linevel.split(), dtype=np.float)
        values = linevel.split()
        # print values,data[0][istep]
        if len(values):
            if (data['step'][istep] != int(values[0])):
                print(data['step'][istep], int(values[0]))
                raise RuntimeError("Different timesteps between files of velocity and thermo")
            for iatom in range(natoms):
                linevel = filevel.readline()
                values = np.array(linevel.split())
                data['vel'][istep, iatom, :] = values[:]

        # lettura forza
        if read_force:
            values = linefor.split()
            # values = np.array(linefor.split(), dtype=np.float)
            # print values,data[0][istep]
            if len(values):
                if (data['step'][istep] != int(values[0])):
                    print(data['step'][istep], int(values[0]))
                    raise RuntimeError("Different timesteps between files of forces and thermo")
                for iatom in range(natoms):
                    linefor = filefor.readline()
                    values = np.array(linefor.split())
                    data['for'][istep, iatom, :] = values[:]

        # lettura stress
        if read_stress:
            # values = np.array(linestr.split(), dtype=np.float64)
            values = linestr.split()
            # print values,data[0][istep]
            if len(values):
                if (data['step'][istep] != int(values[0])):
                    print(data['step'][istep], int(values[0]))
                    raise RuntimeError("Different timesteps between files of stress and thermo")
                for iiline in range(3):
                    linestr = filestr.readline()
                    values = np.array(linestr.split())
                    data['str'][istep, 3 * iiline:3 * iiline + 3] = values[:]

        # lettura cella
        # values = np.array(linecel.split(), dtype=np.float64)
        values = linecel.split()
        # print values, data['step'][istep]
        if len(values):
            if (data['step'][istep] != int(values[0])):
                print(data['step'][istep], int(values[0]))
                raise RuntimeError("Different timesteps between files of cell and thermo")
            for i in range(3):
                values = np.array(filecel.readline().split())
                data['cell'][istep, 3 * i] = values[0]
                data['cell'][istep, 3 * i + 1] = values[1]
                data['cell'][istep, 3 * i + 2] = values[2]

        istep += 1
    return data


def write_xyz(outfile, data, natoms_per_type, type_names=None, type_array=None, xyz=False, vel=False, charge=None,
              tskip=1, vcm=False, raw=False, shuffle=False, poscar=False, nposcar=1):
    """
    Scrive un file nel formato lammpstrj (facilmente leggibile da vmd).
    cp.x nell'output separa gli atomi per tipi. Questa funzione assume lo stesso ordine.
    outfile is il nome del file da scrivere.
    data is il risultato della chiamata a read_file_pos_vel
    l is la dimensione della cella cubica scritta nell'output. """

    ## Conversion factors
    conv_pos = BOHR
    conv_vel = BOHR / TAU
    conv_for = HARTREE / BOHR
    conv_energy = HARTREE
    conv_virial = bar

    ## Put data in variables: improve readability
    POS = data['pos'] * conv_pos
    VEL = data['vel'] * conv_vel
    if 'for' in data: FOR = data['for'] * conv_for
    CELL = data['cell'] * conv_pos
    STEP = data['step']
    TEMP = data['Tion']
    EPOT = data['epot'] * conv_energy
    if 'str' in data: VIR = data['str'] * conv_virial

    out_file = open(outfile, "w")

    if xyz:
        out_xyz = open(prefix + '.xyz', 'w')

    if analisi:
        out_anal = open(prefix + '.analisi', 'w')

    if charge is not None:
        out_j = open(prefix + '.current', 'w')
        out_j.write('step   Temp   c_jion[1]   c_jion[2]   c_jion[3] # e*Ang/ps\n')

    if vcm:
        out_vcm = []
        for ityp, typ in enumerate(species):
            out_vcm.append(open(prefix + '.{}.vcm'.format(typ), 'w'))
            out_vcm[ityp].write(
                'step   Temp   c_vcm{spec:s}[1]   c_vcm{spec:s}[2]   c_vcm{spec:s}[3] # Ang/ps\n'.format(spec=typ))

    if raw:
        out_box_raw = open('box.raw', 'w')
        out_coord_raw = open('coord.raw', 'ba')
        out_coord_raw.truncate(0)
        if 'for' in data:
            out_force_raw = open('force.raw', 'ba')
            out_force_raw.truncate(0)
        out_energy_raw = open('energy.raw', 'w')
        if 'str' in data:
            out_virial_raw = open('stress.raw', 'ba')
            out_virial_raw.truncate(0)
        if 'vel' in data:
            out_vel_raw = open('vel.raw', 'ba')
            out_vel_raw.truncate(0)

    # out_file.write("This Text is going to out file\nLook at it and see\n")
    nsteps = POS.shape[0]
    natoms = POS.shape[1]
    if (natoms != sum(natoms_per_type)):
        raise ValueError('Sum of number of atoms per type does not match the total number of atoms.')
    if type_names is None:
        type_names = map(str, np.arange(1, len(natoms_per_type) + 1))
    else:
        if (len(natoms_per_type) != len(type_names)):
            raise ValueError('Number of type_names not compatible with natoms_per_type.')

    if vel:
        np.savetxt(prefix + '.atmvel', np.reshape(VEL, (nsteps, 3 * natoms)))

    # if needed, shuffle data for 'raw' files generation
    steps_raw = np.arange(0, nsteps, dtype=np.int)
    if shuffle:
        np.random.shuffle(steps_raw)

    if poscar:
        os.mkdir('FOLDER_POSCAR')

    for itimestep in range(0, nsteps, tskip):

        if xyz:
            out_xyz.write('{}\n\n'.format(natoms))

        if analisi:
            out_anal.write('{}\n'.format(natoms))
            out_anal.write('{} {}\n'.format(0, CELL[itimestep, 0]))
            out_anal.write('{} {}\n'.format(0, CELL[itimestep, 4]))
            out_anal.write('{} {}\n'.format(0, CELL[itimestep, 8]))

        out_file.write("ITEM: TIMESTEP\n")
        out_file.write("{}\n".format(int(round(STEP[itimestep]))))
        out_file.write("ITEM: NUMBER OF ATOMS\n")
        out_file.write("{}\n".format(natoms))
        out_file.write('ITEM: BOX BOUNDS pp pp pp\n')
        out_file.write('{} {}\n'.format(0, CELL[itimestep, 0]))
        out_file.write('{} {}\n'.format(0, CELL[itimestep, 4]))
        out_file.write('{} {}\n'.format(0, CELL[itimestep, 8]))
        out_file.write('ITEM: ATOMS id type x y z vx vy vz\n')
        cumnattype = np.cumsum(np.append(0, natoms_per_type))

        jion = np.zeros(3, dtype=np.float)

        if vcm:
            vcom = np.zeros((len(species), 3), dtype=np.float)
            vcom2 = np.zeros((len(species), 3), dtype=np.float)

        if raw:
            itimestep_raw = steps_raw[itimestep]
            # generate, once and for all, the type.raw file
            if itimestep == 0:
                with open('type.raw', 'w') as f:
                    for iatom, attype in enumerate(type_array):
                        f.write('{} '.format(attype))
            # out_coord_raw.write(np.reshape(POS[itimestep_raw, :, :], 3*natoms))
            # np.savetxt(out_coord_raw, np.reshape(POS[itimestep_raw, :, :], 3*natoms), newline = " ")
            tosave = POS[itimestep_raw, :, :]
            sides = np.diag(np.reshape(CELL[itimestep_raw], (3, 3)))
            tosave = np.reshape(tosave % sides, 3 * natoms)  # TODO: implement PBC in more general cases
            np.savetxt(out_coord_raw, tosave, newline=" ")
            # print(tosave)
            out_coord_raw.write('\n'.encode("utf-8"))
            # if 'for' in data: out_force_raw.write(np.reshape(FOR[itimestep_raw, :, :], 3*natoms))
            if 'for' in data:
                np.savetxt(out_force_raw, np.reshape(FOR[itimestep_raw, :, :], 3 * natoms), newline=" ")
                out_force_raw.write('\n'.encode("utf-8"))
            if 'str' in data:
                np.savetxt(out_virial_raw, np.reshape(VIR[itimestep_raw, :], 9), newline=" ")
                out_virial_raw.write('\n'.encode("utf-8"))
            # out_box_raw.write(np.reshape(CELL[itimestep_raw, :, :], 9*natoms))
            if 'vel' in data:
                np.savetxt(out_vel_raw, np.reshape(VEL[itimestep_raw, :, :], 3 * natoms), newline=" ")
                out_vel_raw.write('\n'.encode("utf-8"))
            for ibox in range(8):
                out_box_raw.write('{} '.format(CELL[itimestep_raw, ibox]))
            out_box_raw.write('{}\n'.format(CELL[itimestep_raw, -1]))
            out_energy_raw.write('{}\n'.format(EPOT[itimestep_raw]))

        for idat, attype in enumerate(type_array):
            out_file.write('{} {} {} {} {} {} {} {}\n'.format(idat + 1, type_names[attype], \
                                                              POS[itimestep, idat, 0],
                                                              POS[itimestep, idat, 1],
                                                              POS[itimestep, idat, 2],
                                                              VEL[itimestep, idat, 0],
                                                              VEL[itimestep, idat, 1],
                                                              VEL[itimestep, idat, 2] \
                                                              ))
            if xyz:
                if vel:
                    out_xyz.write('{:s} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n'.format(
                        type_names[attype],
                        POS[itimestep, idat, 0],
                        POS[itimestep, idat, 1],
                        POS[itimestep, idat, 2],
                        VEL[itimestep, idat, 0],
                        VEL[itimestep, idat, 1],
                        VEL[itimestep, idat, 2] \
                        ))
                else:
                    out_xyz.write('{:s} {:15.10f} {:15.10f} {:15.10f}\n'.format(
                        type_names[attype],
                        POS[itimestep, idat, 0],
                        POS[itimestep, idat, 1],
                        POS[itimestep, idat, 2] \
                        ))
            if analisi:
                out_anal.write(
                    '{:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n'.format(idat + 1,
                                                                                                               attype, \
                                                                                                               POS[
                                                                                                                   itimestep, idat, 0],
                                                                                                               POS[
                                                                                                                   itimestep, idat, 1],
                                                                                                               POS[
                                                                                                                   itimestep, idat, 2],
                                                                                                               VEL[
                                                                                                                   itimestep, idat, 0],
                                                                                                               VEL[
                                                                                                                   itimestep, idat, 1],
                                                                                                               VEL[
                                                                                                                   itimestep, idat, 2] \
                                                                                                               ))

            if charge is not None:
                jion += charge[attype] * VEL[itimestep, idat, :]

            if vcm:
                vcom[attype, :] += VEL[itimestep, idat, :]
                vcom2[attype, :] += VEL[itimestep, idat, :] ** 2

        if vcm:
            for typ, typname in enumerate(species):
                out_vcm[typ].write(
                    '{:d} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n'.format(
                        int(round(STEP[itimestep])),
                        TEMP[itimestep],
                        vcom[typ][0],
                        vcom[typ][1],
                        vcom[typ][2],
                        vcom2[typ][0],
                        vcom2[typ][1],
                        vcom2[typ][2] \
                        ))

        if charge is not None:
            out_j.write('{:d} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n'.format(int(round(STEP[itimestep])),
                                                                                TEMP[itimestep],
                                                                                jion[0],
                                                                                jion[1],
                                                                                jion[2] \
                                                                                ))
        if poscar:
            if itimestep % (nsteps//nposcar):
                with open('FOLDER_POSCAR/POSCAR.{}'.format(itimestep // (nsteps//nposcar)), 'w+') as g:
                    for itype in range(natoms_per_type):
                        g.write('{}'.format(type_names[itype])+'{} '.format(natoms_per_type[itype]))
                    g.write('\n')
                    g.write('1.0\n')
                    g.write('{} {}\n'.format(0, CELL[itimestep, 0]))
                    g.write('{} {}\n'.format(0, CELL[itimestep, 4]))
                    g.write('{} {}\n'.format(0, CELL[itimestep, 8]))
                    for itype in range(natoms_per_type):
                        g.write('{} '.format(type_names[itype]))
                    g.write('\n')
                    for itype in range(natoms_per_type):
                        g.write('{} '.format(natoms_per_type[itype]))
                    g.write('\n')
                    g.write('Cartesian\n')
                    for itype in range(natoms_per_type):
                        listat=np.where(type_array==itype)[0]
                        for idat in listat:
                            g.write(' {} {} {} \n'.format(POS[itimestep, idat, 0],
                                                          POS[itimestep, idat, 1],
                                                          POS[itimestep, idat, 2] \
                                                          ))

    if raw:
        if 'vel' in data:
            out_vel_raw.close()
        if 'str' in data:
            out_virial_raw.close()
        if 'for' in data:
            out_force_raw.close()
        out_coord_raw.close()
        out_energy_raw.close()
        out_box_raw.close()
    out_file.close()
    return


###########################################################################################################################################################################
### Parser

parser = argparse.ArgumentParser(
    description='Convert a cp.x trajectory file to the LAMMPS trajectory format. The units are Angstrom and Angstrom/picosecond.')
parser.add_argument('-d', '--directory',
                    type=str,
                    required=False,
                    help='Directory with the .pos and .vel files.',
                    default='./tmp')
parser.add_argument('-p', '--prefix',
                    type=str,
                    required=True,
                    help='Prefix of the filename.')
parser.add_argument('-s', '--species',
                    nargs='*',
                    type=str,
                    required=True,
                    help='Sequence of atomic species in the simulation (in the same order as in the ATOMIC_SPECIES card in the cp.x input).')
parser.add_argument('-n', '--natm',
                    nargs='*',
                    type=int,
                    required=True,
                    help='Number of atoms per species (in the same order as in the ATOMIC_SPECIES card in the cp.x input).')
parser.add_argument('-c', '--charge',
                    nargs='*',
                    type=float,
                    required=False,
                    help='Oxidation number per species (in the same order as in the ATOMIC_SPECIES card in the cp.x input).')
parser.add_argument('--nstep',
                    type=int,
                    default=None,
                    help='Number of steps to convert.')
parser.add_argument('--tskip',
                    type=int,
                    default=1,
                    help='Write 1 every tskip steps.')
parser.add_argument('--xyz',
                    help='Write the coordinates in a .xyz file.',
                    action='store_true',
                    required=False,
                    default=False)
parser.add_argument('--vel',
                    help='Write also the velocities in a .xyz file.',
                    action='store_true',
                    required=False,
                    default=False)
parser.add_argument('--analisi',
                    help='Output data in analisi format.',
                    action='store_true',
                    default=False)
parser.add_argument('--vcm',
                    help='Write the per-species velocity of the centre of mass.',
                    action='store_true',
                    required=False,
                    default=False)
parser.add_argument('--raw',
                    help='Write the .raw files needed for deepMD.',
                    action='store_true',
                    required=False,
                    default=False)
parser.add_argument('--shuffle',
                    help='In writing the .raw files needed for deepMD, shuffle the steps.',
                    action='store_true',
                    required=False,
                    default=False)
parser.add_argument('--poscar',
                    help='Write the POSCAR files needed for deepMD.',
                    action='store_true',
                    required=False,
                    default=False)

parser.add_argument('--nposcar',
                    help='Write the NPOSCAR POSCAR files needed for deepMD.',
                    action='store_true',
                    required=False,
                    default=10)


args = parser.parse_args()

directory = args.directory
prefix = args.prefix
species = args.species
natm = args.natm
nstep = args.nstep
xyz = args.xyz
vel = args.vel
analisi = args.analisi
charge = args.charge
tskip = args.tskip
vcm = args.vcm
raw = args.raw
shuffle = args.shuffle
poscar = args.poscar
nposcar = args.nposcar

if shuffle:
    if not raw:
        raise ValueError('--shuffle is only for --raw output.')

if isinstance(species, list):
    if not isinstance(natm, list):
        raise ValueError('--natm should have the same dimension of --species!')
    else:
        if len(species) != len(natm):
            raise ValueError('--natm should have the same dimension of --species!')
if isinstance(natm, list):
    if not isinstance(species, list):
        raise ValueError('--natm should have the same dimension of --species!')
    else:
        if len(species) != len(natm):
            raise ValueError('--natm should have the same dimension of --species!')

if isinstance(natm, list):
    natm_tot = np.sum(natm)
else:
    natm_tot = natm

print('Reading {}/{}...'.format(directory, prefix))
type_array = read_input_type(prefix, natm, species)
leggi = read_file_pos_vel('{}/{}'.format(directory, prefix), natm_tot, nstep=nstep)
print('Done.')
print('Writing output files...')
scrivi = write_xyz('{}.lammpstrj'.format(prefix), leggi, natm, species, type_array, xyz=xyz, vel=vel, charge=charge,
                   tskip=tskip, vcm=vcm, raw=raw, shuffle=shuffle, poscar=poscar, nposcar=nposcar)
print('Done.')
