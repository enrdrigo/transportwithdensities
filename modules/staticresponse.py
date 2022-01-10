from modules import initialize
from modules import compute
import matplotlib.pyplot as plt
import numpy as np

root = './'
filename = 'dump1.1fs.lammpstrj'

posox = 0.125
nkpoints = 10
ntrysnap = -1
inputcompute = initialize.getinitialize(filename, root, posox, nkpoints, ntrysnap)

temp = 300

staticresponse = compute.computestaticresponse(inputcompute['root'], inputcompute['filename'], inputcompute['N'], \
                                             inputcompute['size'], inputcompute['position of the ox'], \
                                             inputcompute['number of k'], inputcompute['number of snapshots'], temp)

