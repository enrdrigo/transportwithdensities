import os
from modules import traj
from modules import initialize

root = './'
filename = 'dump1.1fs.lammpstrj'
fileinit = 'init.dat'

posox = float(input('position of the oxy:>\n'))
nkpoints = 100
ntrysnap = -1
if os.path.exists(root+filename):
    inputcompute = initialize.getinitialize(filename, root, posox, nkpoints, ntrysnap)
else:
    inputcompute = initialize.getinitialize(fileinit, root, posox, nkpoints, ntrysnap)


trajnew.read_dump(inputcompute['root'], inputcompute['filename'],
          inputcompute['N'], inputcompute['number of snapshots'])

natpermol = int(input('number of atoms per molecule:>\n'))

trajnew.computekftnumba(inputcompute['root'], inputcompute['N'],
                               inputcompute['size'], inputcompute['position of the ox'],
                               inputcompute['number of k'], inputcompute['number of snapshots'], natpermol)
