from modules import initialize
from modules import computestaticresponse
import os

root = './'
filename = 'dump1.1fs.lammpstrj'
fileinit = 'init.dat'

posox = float(input('position of the oxy:>'))
nkpoints = 100
ntrysnap = -1
if os.path.exists(root+filename):
    inputcompute = initialize.getinitialize(filename, root, posox, nkpoints, ntrysnap)
else:
    inputcompute = initialize.getinitialize(fileinit, root, posox, nkpoints, ntrysnap)

temp = float(input('temperature:>'))

natpermol = int(input('number of atoms per molecule:>'))

staticresponse = computestaticresponse.computestaticresponse(inputcompute['root'], inputcompute['size'],
                                                             inputcompute['number of k'], temp)

