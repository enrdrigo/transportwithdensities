import numpy as np
import pickle as pkl
from modules import molar
from modules import tools
from modules import initialize
import os


def spcecificheat(root, filename, filename_loglammps, nk, posox, UNITS):
    inp = initialize.getinitialize(filename, root, posox, nk, -1)
    print('c_p and c_v in units of J/mol/K')
    if os.path.exists(root + filename_loglammps + '.npy'):
        s = np.load(root + filename_loglammps + '.npy', allow_pickle='TRUE').item()
    else:
        s = molar.read_log_lammps(root=root, filename=filename_loglammps)
    with open(root + 'n1k.pkl', 'rb') as f:
        n1k = np.array(pkl.load(f)).T[:, :]
    with open(root + 'n2k.pkl', 'rb') as f:
        n2k = np.array(pkl.load(f)).T[:, :]
    with open(root + 'enk.pkl', 'rb') as f:
        enk = np.array(pkl.load(f)).T[:, :]

    h = molar.molar_enthalpy(root, filename, filename_loglammps, inp['size'].prod(), inp['N'], 12, UNITS=UNITS)
    hm = h.mean(axis=1).mean(axis=0)

    L = inp['size']
    if UNITS == 'real':
        fac = (4186 / 6.022e23) ** 2 / s['Temp'].mean() ** 2 / 1.38e-23 * 6.022e23 / inp['N']
    elif UNITS == 'metal':
        fac = (1.60218e-19) ** 2 / s['Temp'].mean() ** 2 / 1.38e-23 * 6.022e23 / inp['N']
    else:
        raise ValueError('NOT IMPLEMENTED YET')
    qk = enk - (s['Enthalpy'] - s['TotEng']).mean() / inp['N'] * (n1k + n2k)
    q_penthk = enk - (+hm[0] * n1k + hm[1] * n2k - s['TotEng'].mean() / inp['N'] * (n1k + n2k))
    e_rk = (enk.T - np.mean(enk * (n1k + n2k).conj(), axis=1) / np.mean((n1k + n2k) * (n1k + n2k).conj(), axis=1) * (
                n1k + n2k).T).T
    cpk = fac * (qk * qk.conj()).mean(axis=1)
    cp_peth = fac * (q_penthk * q_penthk.conj()).mean(axis=1)
    cvk = fac * (e_rk * e_rk.conj()).mean(axis=1)
    cek = fac * (enk * enk.conj()).mean(axis=1)

    std = tools.stdblock_parallel((qk * qk.conj()) * fac, ncpus=16)
    stdcp = np.sqrt(std[:, 0, int(std.shape[2] / 2)])
    std = tools.stdblock_parallel((q_penthk * q_penthk.conj()) * fac, ncpus=16)
    stdcp_peth = np.sqrt(std[:, 0, int(std.shape[2] / 2)])
    std = tools.stdblock_parallel((e_rk * e_rk.conj()) * fac, ncpus=16)
    stdcv = np.sqrt(std[:, 0, int(std.shape[2] / 2)])
    std = tools.stdblock_parallel((enk * enk.conj()) * fac, ncpus=16)
    stdce = np.sqrt(std[:, 0, int(std.shape[2] / 2)])

    G = tools.Ggeneratemodall(inp['number of k'], inp['size']) * 2 * np.pi

    res = {}
    res = {'k': G,
          'cp': cpk,
          'std cp': stdcp,
          'cv': cvk,
          'std cv': stdcv,
          'cp with partial enthalpies': cp_peth,
          'std cp with partial enthalpies': stdcp_peth}

    np.save(root+'specieficheat.npy', res)

    with open(inp['root'] + 'cp.out', 'w') as f:
        for i in range(cpk.shape[1] - 1):
            f.write(
                '{}\t'.format(G[i + 1] * 10) + '{}\t'.format(np.real(cpk)[i]) + '{}\n'.format(stdcp[i]))
    with open(inp['root'] + 'cv.out', 'w') as f:
        for i in range(cvk.shape[1] - 1):
            f.write(
                '{}\t'.format(G[i + 1] * 10) + '{}\t'.format(np.real(cvk)[i]) + '{}\n'.format(stdcv[i]))

    with open(inp['root'] + 'cpenth.out', 'w') as f:
        for i in range(cp_peth.shape[1] - 1):
            f.write(
                '{}\t'.format(G[i + 1] * 10) + '{}\t'.format(np.real(cp_peth)[i]) + '{}\n'.format(stdcp_peth[i]))

    return res
