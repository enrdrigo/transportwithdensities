from modules import initialize
from modules import computestaticresponse
import os

root = './'
filename = 'dump1.1fs.lammpstrj'
fileinit = 'init.dat'

with open('param.dat', 'r') as f:
    nkpoints = int(f.readline())
    posox = float(f.readline())
    natpermol = int(f.readline())

ntrysnap = -1

if os.path.exists(root+filename):
    inputcompute = initialize.getinitialize(filename, root, posox, nkpoints, ntrysnap)
else:
    inputcompute = initialize.getinitialize(fileinit, root, posox, nkpoints, ntrysnap)

temp = float(input('temperature:>'))


staticresponse = computestaticresponse.computestaticresponse(inputcompute['root'], inputcompute['size'],
                                                             inputcompute['number of k'], temp)

