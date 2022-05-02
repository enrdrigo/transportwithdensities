from modules import traj
from modules import initialize

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

traj.computekftnumba(inputcompute['root'], inputcompute['N'],
                               inputcompute['size'], inputcompute['position of the ox'],
                               inputcompute['number of k'], inputcompute['number of snapshots'], natpermol)
