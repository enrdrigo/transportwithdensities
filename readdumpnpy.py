from modules import traj
from modules import initialize
from modules import tools

root = './'
filename = str(input('filename:>\n'))
fileinit = 'init.dat'

posox = float(input('position of the oxy:>\n'))
nkpoints = 100
ntrysnap = -1

with open(root+fileinit, 'w') as f:
    with open(root+filename, 'r') as g:
        for i in range(10):
            line = g.readline()
            f.write('{}'.format(line))

inputcompute = initialize.getinitialize(fileinit, root, posox, nkpoints, ntrysnap)

traj.read_dump(inputcompute['root'], filename,
          inputcompute['N'], inputcompute['number of snapshots'])

natpermol = int(input('number of atoms per molecule:>\n'))

with open(root+'param.dat', 'w') as f:
    f.write('{}\n'.format(nkpoints))
    f.write('{}\n'.format(posox))
    f.write('{}\n'.format(natpermol))

dumpreadh5py = tools.readarraydatah5py(root=root,
                                     filename=filename,
                                     posox=posox,
                                     nk=nkpoints
                                    )

enka = traj.computescalarkft_parallel(f=dumpreadh5py['energy density'].sum(axis=2),
                                    x=dumpreadh5py['pos'],
                                    G=dumpreadh5py['G vectors'],
                                    root=root,
                                    outname='enka'
                                   )

enkina = traj.computescalarkft_parallel(f=dumpreadh5py['energy density'][0],
                                    x=dumpreadh5py['pos'],
                                    G=dumpreadh5py['G vectors'],
                                    root=root,
                                    outname='enkina'
                                   )

chka = traj.computescalarkft_parallel(f=dumpreadh5py['charge density'],
                                    x=dumpreadh5py['pos'],
                                    G=dumpreadh5py['G vectors'],
                                    root=root,
                                    outname='chka'
                                   )

n1ka = traj.computescalarkft_parallel(f=dumpreadh5py['number density species 1'],
                                    x=dumpreadh5py['pos'],
                                    G=dumpreadh5py['G vectors'],
                                    root=root,
                                    outname='n1ka'
                                   )

n2ka = traj.computescalarkft_parallel(f=dumpreadh5py['number density species 2'],
                                    x=dumpreadh5py['pos'],
                                    G=dumpreadh5py['G vectors'],
                                    root=root,
                                    outname='n2ka'
                                   )