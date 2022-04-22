import pickle as pk
from modules import computestaticresponse

import matplotlib.pyplot as plt
import sys
import numpy as np
from modules import traj
from modules import initialize
from modules import molar

def stdblock(array):
    var = list()
    binsize = list()
    nbino = 0
    for i in range(1, int(len(array) / 10)):
        nbin = int(len(array) / i)
        if nbin == nbino:
            continue
        rarray = np.reshape(array[:nbin * i], (nbin, i))
        barray = np.zeros(nbin, dtype=np.complex_)
        barray = np.sum(rarray, axis=1) / i
        var.append(np.var(barray) / nbin)
        binsize.append(i)
        nbino = nbin

    return np.array(var), np.array(binsize)

inp=initialize.getinitialize(filename='dumpKCl.lammpstrj',
                             root='/marconi/home/userexternal/edrigo00/KCl/',
                             posox=0.,
                             nk=100,
                             ntry=-1)

G=computestaticresponse.Ggeneratemodall(inp['number of k'], inp['size'])* 2 * np.pi

filename_loglammps='KCl.lammps'

log=molar.read_log_lammps(root=inp['root'],
                          filename=filename_loglammps)

print(log['Temp'].mean(), 'K ', log['Temp'].std()/np.sqrt(len(log['Temp'])), 'K ',
      log['Press'].mean(), 'atm', log['Press'].std()/np.sqrt(len(log['Press'])), 'atm')

hp=molar.molar_enthalpy(root=inp['root'],
                     filename=inp['filename'],
                     filename_log=filename_loglammps,
                     volume=inp['size'].prod(),
                     Np=inp['N'],
                     nbloks=12,
                     UNITS='metal')

h=hp.mean(axis=0).mean(axis=0)

with open(inp['root']+'enk.pkl', 'rb') as g:
    enkb=pk.load(g)
    enk=np.array(enkb)
    enkb=0
with open(inp['root']+'chk.pkl', 'rb') as g:
    chkb=pk.load(g)
    chk=np.array(chkb)
    chkb=0
with open(inp['root']+'n1k.pkl', 'rb') as g:
    n1kb=pk.load(g)
    n1k=np.array(n1kb)
    n1kb=0
with open(inp['root']+'n2k.pkl', 'rb') as g:
    n2kb=pk.load(g)
    n2k=np.array(n2kb)
    n2kb=0



a=np.zeros(chk.shape[1]-1, np.complex_)
b=np.zeros(chk.shape[1]-1, np.complex_)
c=np.zeros(chk.shape[1]-1, np.complex_)
va=np.zeros(chk.shape[1]-1)
vb=np.zeros(chk.shape[1]-1)
vc=np.zeros(chk.shape[1]-1)

fac = (16.022 * 1.0e-30 * 1.60218e-19 * 1.0e-10 /
       (inp['size'].prod() * 1.0e-30 * 1.38e-23 * log['Temp'].mean() ** 2 * 8.854 * 1.0e-12))

a=(np.mean((enk)*np.conj(chk), axis=0)/G**2)[1:]*fac
b=-(np.mean((n1k*h[0]+h[1]*n2k)*np.conj(chk), axis=0)/G**2)[1:]*fac
for i in range(1,chk.shape[1]):
    std, bins=stdblock(fac*(enk[:,i])*np.conj(chk[:, i])/G[i]**2)
    pp = int(16 * len(std) / 20)
    va[i-1] = std[pp]
    std, bins=stdblock(fac*(n1k[:,i]*h[0]+h[1]*n2k[:,i])*np.conj(chk[:,i])/G[i]**2)
    pp = int(16 * len(std) / 20)
    vb[i-1] = std[pp]
with open('skcl.out', 'w') as f:
    for i in range(chk.shape[1]-1):
        f.write('{}\t'.format(G[i+1]*10)+'{}\t'.format(np.real(a+b)[i])+'{}\n'.format(np.sqrt(va+vb)[i]))

f, ax=plt.subplots(1)
ax.errorbar(G[1:]*10, a+b, np.sqrt(va+vb), fmt='.')

f,ax =plt.subplots(3, figsize=[8,12])

ax[0].errorbar(G[1:]*10,b, np.sqrt(vb), fmt='.', label='contributo delle entalpie')
ax[0].legend()
ax[0].set_xlabel(r'k [$nm^{-1}$]')
ax[0].set_ylabel(r'S_enth [$\frac{V}{K}$]')
ax[1].errorbar(G[1:]*10,a, np.sqrt(va), fmt='.', label='contributo delle energie')
ax[1].legend()
ax[1].set_xlabel(r'k [$nm^{-1}$]')
ax[1].set_ylabel(r'S_en [$\frac{V}{K}$]')
ax[2].errorbar(G[1:]*10,a+b, np.sqrt(va+vb), fmt='.', label='totale')
ax[2].legend()
ax[2].set_xlabel(r'k [$nm^{-1}$]')
ax[2].set_ylabel(r'S [$\frac{V}{K}$]')


with open('skcl.out', 'w') as f:
    for i in range(chk.shape[1]-1):
        f.write('{}\t'.format(G[i+1]*10)+'{}\t'.format(np.real(a+b)[i])+'{}\n'.format(np.sqrt(va+vb)[i]))