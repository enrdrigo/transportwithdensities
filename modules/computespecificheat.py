import numpy as np
import pickle as pkl
from modules import molar
from modules import tools
from modules import initialize
import os


def spcecificheat(root, filename, filename_loglammps, nk, posox, UNITS, enthalpy=True):
    inp = initialize.getinitialize(filename, root, posox, nk, -1)
    print('c_p and c_v in units of J/mol/K')
    if os.path.exists(root + filename_loglammps + '.npy'):
        s = np.load(root + filename_loglammps + '.npy', allow_pickle='TRUE').item()
    else:
        s = molar.read_log_lammps(root=root, filename=filename_loglammps)

    try:
        enk = np.load(inp['root'] + 'enka.npy').T
        chk = np.load(inp['root'] + 'chka.npy').T
        n1k = np.load(inp['root'] + 'n1ka.npy').T
        n2k = np.load(inp['root'] + 'n2k.npy').T
    except:
        with open(inp['root'] + 'enk.pkl', 'rb') as g:
            enkb = pkl.load(g)
            enk = np.array(enkb).T
            enkb = 0
        with open(inp['root'] + 'chk.pkl', 'rb') as g:
            chkb = pkl.load(g)
            chk = np.array(chkb).T
            chkb = 0
        with open(inp['root'] + 'n1k.pkl', 'rb') as g:
            n1kb = pkl.load(g)
            n1k = np.array(n1kb).T
            n1kb = 0
        with open(inp['root'] + 'n2k.pkl', 'rb') as g:
            n2kb = pkl.load(g)
            n2k = np.array(n2kb).T
            n2kb = 0

    #with open(root + 'n1k.pkl', 'rb') as f:
    #    n1k = np.array(pkl.load(f)).T[:, :]
    #with open(root + 'n2k.pkl', 'rb') as f:
    #    n2k = np.array(pkl.load(f)).T[:, :]
    #with open(root + 'enk.pkl', 'rb') as f:
    #    enk = np.array(pkl.load(f)).T[:, :]

    if enthalpy:
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
    if enthalpy:
        q_penthk = enk - (+hm[0] * n1k + hm[1] * n2k - s['TotEng'].mean() / inp['N'] * (n1k + n2k))
    e_rk = (enk.T - np.mean(enk * (n1k + n2k).conj(), axis=1) / np.mean((n1k + n2k) * (n1k + n2k).conj(), axis=1) * (
                n1k + n2k).T).T
    cpk = fac * (qk * qk.conj()).mean(axis=1)
    if enthalpy:
        cp_peth = fac * (q_penthk * q_penthk.conj()).mean(axis=1)
    cvk = fac * (e_rk * e_rk.conj()).mean(axis=1)
    cek = fac * (enk * enk.conj()).mean(axis=1)

    std = tools.stdblock_parallel((qk * qk.conj()) * fac, ncpus=16)
    stdcp = np.sqrt(std[:, 0, int(std.shape[2] / 2)])
    if enthalpy:
        std = tools.stdblock_parallel((q_penthk * q_penthk.conj()) * fac, ncpus=16)
        stdcp_peth = np.sqrt(std[:, 0, int(std.shape[2] / 2)])
    std = tools.stdblock_parallel((e_rk * e_rk.conj()) * fac, ncpus=16)
    stdcv = np.sqrt(std[:, 0, int(std.shape[2] / 2)])
    std = tools.stdblock_parallel((enk * enk.conj()) * fac, ncpus=16)
    stdce = np.sqrt(std[:, 0, int(std.shape[2] / 2)])

    G = tools.Ggeneratemodall(inp['number of k'], inp['size']) * 2 * np.pi

    res = {}
    if enthalpy:
        res = {'k': G,
              'cp': cpk,
              'std cp': stdcp,
              'cv': cvk,
              'std cv': stdcv,
              'cp with partial enthalpies': cp_peth,
              'std cp with partial enthalpies': stdcp_peth}
    else:
        res = {'k': G,
               'cp': cpk,
               'std cp': stdcp,
               'cv': cvk,
               'std cv': stdcv}

    np.save(root+'specieficheat.npy', res)

    with open(inp['root'] + 'cp.out', 'w') as f:
        for i in range(cpk.shape[0] - 1):
            f.write(
                '{}\t'.format(G[i + 1] * 10) + '{}\t'.format(np.real(cpk)[i+1]) + '{}\n'.format(stdcp[i+1]))
    with open(inp['root'] + 'cv.out', 'w') as f:
        for i in range(cvk.shape[0] - 1):
            f.write(
                '{}\t'.format(G[i + 1] * 10) + '{}\t'.format(np.real(cvk)[i+1]) + '{}\n'.format(stdcv[i+1]))
    if enthalpy:
        with open(inp['root'] + 'cpenth.out', 'w') as f:
            for i in range(cp_peth.shape[0] - 1):
                f.write(
                    '{}\t'.format(G[i + 1] * 10) + '{}\t'.format(np.real(cp_peth)[i+1]) + '{}\n'.format(stdcp_peth[i+1]))

    return res
