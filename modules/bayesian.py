import numpy as np
from numpy.linalg import eig
from . import cubicharmonics
import logging
import copy

def generatesorteddata(data, nk):
    G = cubicharmonics.Gvecgenerateall(nk)

    gmod = np.linalg.norm(G[1:], axis=1)

    dicdata = {}
    dicg = {}
    sdata = np.zeros((3, len(data[0])))
    nr = np.random.random(len(data[0])) * 1.0e-6
    for i in range(len(data[0])):
        dicdata[gmod[i] + nr[i]] = [data[1][i], data[2][i]]
        dicg[gmod[i] + nr[i]] = [G[i + 1][0], G[i + 1][1], G[i + 1][2]]
    data0sort = np.sort(gmod[:] + nr)

    sdata = np.zeros((2, len(data[0])))
    grid = np.zeros((len(data[0]), 3))
    for i in range(len(data[0])):
        grid[i][0] = dicg[data0sort[i]][0]
        grid[i][1] = dicg[data0sort[i]][1]
        grid[i][2] = dicg[data0sort[i]][2]
        sdata[0][i] = dicdata[data0sort[i]][0]
        sdata[1][i] = dicdata[data0sort[i]][1]

    return sdata, grid


def datainit(root, filename, nk):
    data = np.transpose(np.loadtxt(root + filename))

    sdata, grid = generatesorteddata(data, nk)

    C = cubicarray(list(grid), pr=False)

    dic, dics = datadicG(data[1], data[2], cubicharmonics.Gvecgenerateall(100)[1:])

    gplot, dataplot, datasigmaplot = datiplot(C, dic, dics)

    sd = np.array([dataplot, datasigmaplot])

    k_min = data[0][0]

    return data, grid, gplot, dataplot, datasigmaplot, sd, k_min


def convergence(list, tr=1):
    for i in range(2, len(list)):
        if abs((list[i] - list[i - 1]) / list[i - 1]) * 100 < tr:
            return i


def opitmalpredictiondataset(root, filename, nk, tr=1, plot=False):

    N_max = 20
    N_iter = 1
    predb = []
    spredb = []
    Npointsb = []
    degreeb = []
    ev_maxb = []
    for N in range(2, N_max, N_iter):

        try:
            mN, SN, y_infer_, sy_infer_, spar, log_evidence_vP_, mv = bestfitdevel(root=root, filename=filename, nk=nk, N=N,
                                                                                   plot=plot)
        except ValueError:
            print('NOPE')
            continue
        degreeb.append(mv)
        predb.append(mN[0])
        spredb.append(SN[0, 0])

        ev_maxb.append(max(log_evidence_vP_ / np.array(log_evidence_vP_).sum()))
    index = convergence(ev_maxb, tr=tr)
    print(index, 'index of the best seebeck prediction')
    return index, predb[index], np.sqrt(spredb[index]), degreeb[index]


def bayesianpol(grid, sdata, M, N, alpha,  x_infer, bethapar=1,  ifprint=False, ifwarning=True, nLbp=0):
    # grid e' la griglia di punti k.
    # data sono i valori calcolati nella simualzione con la std dev dei dati.
    # M e' il grado massimo dei polinomi che considero.
    # N e' il numero di dati nel fit.
    # alpha e' il parametro di regolarizzazione.
    # x_infer sono i punti k dove voglio inferire il risultato.
    sigma_noise = sdata[1][:N]
    if ifwarning:
        logging.warning(str('HO IMPOSTATO A MANO CHE 2\PI/L = 0.20230317263794637'))
    x = grid[:N, :].T
    #x_infer = grid[:N, :].T * 0.13484487571168569
    y_noise = sdata[0][:N]
    betha = bethapar * (1 / sigma_noise) ** 2

    Phi, contanumpol = computephicubichandL(x, betha, M, nL=nLbp)

    SN_inv = alpha * np.identity(contanumpol) + np.dot(Phi, Phi.T)
    SN = np.linalg.inv(SN_inv)
    mN = np.dot(np.dot(SN, Phi), y_noise[:] * np.sqrt(betha))

    if ifprint: print('parametri ottimali', mN)
    if ifprint: print('numero di armoniche cubiche', contanumpol)

    Phi_infer, contanumpolinfer = computephicubichandL(x_infer, np.ones(np.shape(x_infer)[1]), M, nL=nLbp)

    y_infer = np.dot(mN, Phi_infer)

    sy_infer = 1/bethapar + np.diag(np.dot(Phi_infer.T, np.dot(SN, Phi_infer)))

    if ifprint: print('valore a kmin inferito', y_infer[0], 'bias', mN[0], 'dato a kmin', sdata[0][0])
    if ifprint: print('determinante matrice delle armoniche cubiche ridotte:', np.linalg.det(np.dot(Phi, Phi.T)))

    return mN, SN, y_infer, sy_infer, contanumpol


def bestfitdevel(root, filename, nk, N, ifbetha=False, ifprintbestfit=False, ifprintfinal=False, nLbf=0, plot=False):
    # grid e' la griglia di punti k.
    # sdata sono i valori calcolati nella simualzione con la std dev dei dati.
    # N e' il numero di dati nel fit.
    # x_infer sono i punti k dove voglio inferire il risultato.

    data, oldgrid, grid, dataplot, datasigmaplot, sdata, k_min = datainit(root=root, filename=filename, nk=nk)

    gplot_0 = np.copy(grid)
    gplot_0[0] = np.zeros(3) + 0.0001
    gplot_0 = (np.ones((100, 3)).T * np.linspace(0.01, np.linalg.norm(grid[N]), 100) / np.sqrt(3)).T

    x_infer = gplot_0.T

    M_tot = 15
    # M_tot e' il numero massimo del grado del polinomio che considero
    log_evidence_vP = []
    alpha_vP = []
    betha_vP = []
    g_vP = np.zeros((M_tot))
    x = grid[:N, :].T*k_min
    # x_infer = grid[:N, :].T * 0.13484487571168569
    y_noise = sdata[0][:N]
    Mv_list = []
    sigma_noise = sdata[1][:N]
    betha0 = (1 / sigma_noise) ** 2
    for M_v in range(1, M_tot):  # number of parameters

        # calcolo il set di funzioni di base (armoniche cubiche) associate a questo grado M_v
        Phi_vP, contanumpol = computephicubichandL(x, betha0, M_v, nL=nLbf)

        # calcolo gli autovalori di Phi_vP, servono per la stima di alpha ottimale. Sono gli autovalori di
        li_vP0, ei_vP = eig(np.dot(Phi_vP, Phi_vP.T))
        # salto quando il determinante della matrice delle armiche cubiche ridotte e' troppo piccolo
        if abs(np.prod(li_vP0)) < 1.0e-100:
            if ifprintbestfit: print('determinante della martice delle armoniche cubiche minore di 1.0e-5, salto')
            continue

        bethap0 = 1
        alpha0 = 1
        delta_alphaP = 1
        delta_alphaP = 1
        alphaP = alpha0
        bethaP = bethap0
        conta = 0

        # inizio il ciclo self-consistente per ottenere il valore migliore di alpha
        while abs(delta_alphaP / (alphaP + 0.1)) > 1e-10 and conta < 1.0e3:
            conta += 1
            li_vP = li_vP0 * bethaP
            SN_vP = np.linalg.inv(alphaP * np.identity(contanumpol) + bethaP * np.dot(Phi_vP, Phi_vP.T))
            mN_vP = bethaP * np.dot(np.dot(SN_vP, Phi_vP), y_noise * np.sqrt(betha0))
            g_vP = np.sum(li_vP.real / (alphaP + li_vP.real))
            alpha1P = g_vP / (np.dot(mN_vP.T, mN_vP))
            betha1P = 1 / (1 / (N - g_vP) * np.sum((y_noise * np.sqrt(betha0) - np.dot(mN_vP, Phi_vP)) ** 2))
            delta_alphaP = alpha1P - alphaP
            delta_bethaP = betha1P - bethaP
            alphaP = alpha1P
            if ifbetha: bethaP = betha1P

        if (abs(delta_alphaP / (alphaP + 0.1)) > 1e-10):
            if ifprintbestfit: print('no convergence', N, x[-1], conta, delta_alphaP, alphaP, M_v)
            continue

        Mv_list.append(M_v)
        alpha_vP.append(alphaP)
        betha_vP.append(bethaP)

        # mi preparo a calcolare la funzione di evidence per il valore ottimale di alpha
        A_vP = alphaP * np.identity(contanumpol) + bethaP * np.dot(Phi_vP, Phi_vP.T)
        E_mNs_vP = bethaP / 2 * (y_noise * np.sqrt(betha0) - np.dot(Phi_vP.T, mN_vP.T)) ** 2
        E_mN_vP = E_mNs_vP.sum()
        log_evidence_vP.append(M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(bethaP * betha0)) - \
                               E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        if ifprintbestfit: print('numero di polinomi cubici fino al grado massimo ', 2 * M_v, ':', contanumpol)
        if ifprintbestfit: print('best alpha:', alphaP, 'deltalpha:', delta_alphaP)
        if ifprintbestfit: print('best betha:', bethaP, 'deltbetha:', delta_bethaP)
        if ifprintbestfit: print('logevidence:',
                                 M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(bethaP * betha0)) - \
                                 E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        if ifprintbestfit: print('contributi alla evidence:')
        if ifprintbestfit: print('dalla normalizzazione ', M_v / 2 * np.log(np.abs(alphaP)), \
                                 N / 2 * np.sum(np.log(bethaP * betha0)))
        if ifprintbestfit: print('dalla likelihood:', -E_mN_vP)
        if ifprintbestfit: print('dalla derivata seconda della likelihood (log(det(A))):',
                                 - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        if ifprintbestfit: print('determinante della matrice delle armoniche cubiche ridotte',
                                 np.linalg.det(np.dot(Phi_vP, Phi_vP.T)), '\n')

    # valuto il grado ottimale del polinomio cercando il massimo della funzione di evidence
    index = log_evidence_vP.index(max(log_evidence_vP))

    # calcolo il fit bayesiano per il valore ottimale di alpha e per il grado che massimizza la evidence.
    mN, SN, y_infer, sy_infer, contabest = bayesianpol(grid, sdata, Mv_list[index], N, alpha_vP[index], \
                                                       x_infer, bethapar=betha_vP[index], ifprint=ifprintfinal, \
                                                       ifwarning=False, nLbp=nLbf)

    if ifprintfinal: print('grado ottimale', 2 * (index + 1), 'grado massimo tentato', 2 * (M_tot - 1))
    if ifprintfinal: print('numero di polinomi nella base ottimale: ', contabest, 'numero di dati', N)
    if ifprintfinal: print('best alpha', alpha_vP[index])
    if ifprintfinal: print('best betha', betha_vP[index])

    if plot:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.fill_between(np.linalg.norm(gplot_0[:], axis=1) * k_min,
                        (y_infer + np.sqrt(sy_infer - 1)), (y_infer - np.sqrt(sy_infer - 1)),
                        alpha=0.3, color='red')
        ax.plot(np.linalg.norm(gplot_0[:], axis=1) * k_min, y_infer ,
                '-', alpha=0.5, color='red', label='bayesian regression fit ')
        ax.errorbar(np.linalg.norm(grid, axis=1) * k_min,
                    dataplot , datasigmaplot ,
                    fmt='.', label='data from MD', color='black')
        plt.tight_layout()
        plt.legend()

    return mN, SN, y_infer, sy_infer, SN.diagonal(), log_evidence_vP, Mv_list[index]


def bestfit(grid, sdata, N, x_infer, ifbetha=False, ifprintbestfit=False, ifprintfinal=True, nLbf=0):
    # grid e' la griglia di punti k.
    # sdata sono i valori calcolati nella simualzione con la std dev dei dati.
    # N e' il numero di dati nel fit.
    # x_infer sono i punti k dove voglio inferire il risultato.

    M_tot = 15
    # M_tot e' il numero massimo del grado del polinomio che considero
    log_evidence_vP = []
    alpha_vP = []
    betha_vP = []
    g_vP = np.zeros((M_tot))
    x = grid[:N, :].T
    #x_infer = grid[:N, :].T * 0.13484487571168569
    y_noise = sdata[0][:N]
    Mv_list = []
    sigma_noise = sdata[1][:N]
    betha0 = (1 / sigma_noise) ** 2
    for M_v in range(1, M_tot):  # number of parameters

        # calcolo il set di funzioni di base (armoniche cubiche) associate a questo grado M_v
        Phi_vP, contanumpol = computephicubichandL(x, betha0, M_v, nL=nLbf)

        # calcolo gli autovalori di Phi_vP, servono per la stima di alpha ottimale. Sono gli autovalori di
        li_vP0, ei_vP = eig(np.dot(Phi_vP, Phi_vP.T))
        # salto quando il determinante della matrice delle armiche cubiche ridotte e' troppo piccolo
        if abs(np.prod(li_vP0))<1.0e-100:
            if ifprintbestfit: print('determinante della martice delle armoniche cubiche minore di 1.0e-5, salto')
            continue
        
        bethap0 = 1
        alpha0 = 1
        delta_alphaP = 1
        delta_alphaP = 1
        alphaP = alpha0
        bethaP = bethap0
        conta = 0

        # inizio il ciclo self-consistente per ottenere il valore migliore di alpha
        while abs(delta_alphaP / (alphaP + 0.1)) > 1e-10 and conta < 1.0e3:
            conta += 1
            li_vP = li_vP0 * bethaP
            SN_vP = np.linalg.inv(alphaP * np.identity(contanumpol) + bethaP * np.dot(Phi_vP, Phi_vP.T))
            mN_vP = bethaP * np.dot(np.dot(SN_vP, Phi_vP), y_noise * np.sqrt(betha0))
            g_vP = np.sum(li_vP.real / (alphaP + li_vP.real))
            alpha1P = g_vP / (np.dot(mN_vP.T, mN_vP))
            betha1P = 1 / (1 / (N - g_vP) * np.sum((y_noise * np.sqrt(betha0) - np.dot( mN_vP, Phi_vP)) ** 2))
            delta_alphaP = alpha1P - alphaP
            delta_bethaP = betha1P - bethaP
            alphaP = alpha1P
            if ifbetha: bethaP = betha1P

        if (abs(delta_alphaP / (alphaP + 0.1)) > 1e-10):
            if ifprintbestfit: print('no convergence', N, x[-1], conta, delta_alphaP, alphaP, M_v)
            continue

        Mv_list.append(M_v)
        alpha_vP.append(alphaP)
        betha_vP.append(bethaP)

        # mi preparo a calcolare la funzione di evidence per il valore ottimale di alpha
        A_vP = alphaP * np.identity(contanumpol) + bethaP*np.dot(Phi_vP, Phi_vP.T)
        E_mNs_vP = bethaP / 2 * (y_noise * np.sqrt(betha0) - np.dot(Phi_vP.T, mN_vP.T)) ** 2
        E_mN_vP = E_mNs_vP.sum()
        log_evidence_vP.append(M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(bethaP * betha0)) - \
                               E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        if ifprintbestfit: print('numero di polinomi cubici fino al grado massimo ', 2 * M_v, ':', contanumpol)
        if ifprintbestfit: print('best alpha:', alphaP, 'deltalpha:', delta_alphaP)
        if ifprintbestfit: print('best betha:', bethaP, 'deltbetha:', delta_bethaP)
        if ifprintbestfit: print('logevidence:', M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(bethaP * betha0)) - \
                                 E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        if ifprintbestfit: print('contributi alla evidence:')
        if ifprintbestfit: print('dalla normalizzazione ', M_v / 2 * np.log(np.abs(alphaP)), \
                                 N / 2 * np.sum(np.log(bethaP*betha0)))
        if ifprintbestfit: print('dalla likelihood:', -E_mN_vP)
        if ifprintbestfit: print('dalla derivata seconda della likelihood (log(det(A))):',- 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        if ifprintbestfit: print('determinante della matrice delle armoniche cubiche ridotte', np.linalg.det(np.dot(Phi_vP, Phi_vP.T)),'\n')

    # valuto il grado ottimale del polinomio cercando il massimo della funzione di evidence
    index = log_evidence_vP.index(max(log_evidence_vP))

    # calcolo il fit bayesiano per il valore ottimale di alpha e per il grado che massimizza la evidence.
    mN, SN, y_infer, sy_infer, contabest = bayesianpol(grid, sdata, Mv_list[index], N, alpha_vP[index], \
                                                       x_infer,bethapar=betha_vP[index], ifprint=ifprintfinal,\
                                                       ifwarning=False, nLbp=nLbf)

    if ifprintfinal: print('grado ottimale', 2 * (index+1), 'grado massimo tentato', 2 * (M_tot - 1))
    if ifprintfinal: print('numero di polinomi nella base ottimale: ', contabest, 'numero di dati', N)
    if ifprintfinal: print('best alpha', alpha_vP[index])
    if ifprintfinal: print('best betha', betha_vP[index])

    return mN, SN, y_infer, sy_infer, SN.diagonal(), log_evidence_vP, Mv_list[index]


def bayesianmodelprediction(grid, sdata, N, x_infer, ifprintmodpred=False, ifprintfinal=False, nLbmp=0):
    # grid e' la griglia di punti k.
    # sdata sono i valori calcolati nella simualzione con la std dev dei dati.
    # N e' il numero di dati nel fit.
    # x_infer sono i punti k dove voglio inferire il risultato.

    M_tot = 15
    # M_tot e' il numero massimo del grado del polinomio che considero
    log_evidence_vP = []
    alpha_vP = []
    betha_vP = np.zeros((M_tot))
    g_vP = np.zeros((M_tot))
    x = grid[:N, :].T
    x_infer = grid[:N, :].T 
    y_noise = sdata[0][:N]
    Mv_list = []
    sigma_noise = sdata[1][:N]
    betha0 = (1 / sigma_noise) ** 2
    mNs = 0
    SNs = 0
    mNa = []#np.zeros(M_tot)
    pcont = 0
    for M_v in range(1, M_tot):  # number of parameters

        # calcolo il set di funzioni di base (armoniche cubiche) associate a questo grado M_v
        Phi_vP, contanumpol = computephicubichandL(x, betha0, M_v, nL=nLbmp)

        # calcolo gli autovalori di Phi_vP, servono per la stima di alpha ottimale
        li_vP, ei_vP = eig(np.dot(Phi_vP, Phi_vP.T))

        alpha0 = 1000
        delta_alphaP = 1
        alphaP = alpha0
        conta = 0

        # inizio il ciclo self-consistente per ottenere il valore migliore di alpha
        while abs(delta_alphaP / (alphaP + 0.1)) > 1e-10 and conta < 1.0e3:
            conta += 1
            SN_vP = np.linalg.inv(alphaP * np.identity(contanumpol) + np.dot(Phi_vP, Phi_vP.T))
            mN_vP = np.dot(np.dot(SN_vP, (Phi_vP)), y_noise * np.sqrt(betha0))
            g_vP = np.sum(li_vP.real / (alphaP + li_vP.real))
            alpha1P = g_vP / (np.dot(mN_vP.T, mN_vP))
            delta_alphaP = alpha1P - alphaP
            alphaP = alpha1P

        if (abs(delta_alphaP / (alphaP + 0.1)) > 1e-10 or abs(np.prod(li_vP))<1.0e-5):
            if (abs(delta_alphaP / (alphaP + 0.1)) > 1e-10):
                if ifprintmodpred: print('no convergence', N, x[-1], conta, delta_alphaP, alphaP, M_v)
            if abs(np.prod(li_vP)) < 1.0e-5:
                if ifprintmodpred: print('determinante della martice delle armoniche cubiche ridotte di grado', 2*M_v,\
                                         ' minore di 1.0e-5, salto')
            continue
        else:
            pcont += 1


        Mv_list.append(M_v)
        alpha_vP.append(alphaP)
        # mi preparo a calcolare la funzione di evidence per il valore ottimale di alpha
        A_vP = alphaP * np.identity(contanumpol) + np.dot(Phi_vP, Phi_vP.T)
        E_mNs_vP = 1 / 2 * (y_noise * np.sqrt(betha0) - np.dot(Phi_vP.T, mN_vP.T)) ** 2
        E_mN_vP = E_mNs_vP.sum()
        log_evidence_vP.append(M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(betha0)) - \
                               E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        log_ev = M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(betha0)) - \
                               E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP)))

        if ifprintmodpred: print('numero di polinomi cubici di grado massimo', 2 * M_v, ':', contanumpol)
        if ifprintmodpred: print('best alpha:', alphaP, 'deltalpha:', delta_alphaP)
        if ifprintmodpred: print('logevidence:', M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(betha0)) - \
                                 E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))

        mN, SN, y_infer, sy_infer, contamod = bayesianpol(grid, sdata, M_v, N, alphaP, grid, ifprint=ifprintmodpred, ifwarning=False, nLbp=nLbmp)
        mNs += mN[0] * np.exp(log_ev / log_evidence_vP[0])

        if ifprintmodpred: print('determinante della matrice delle armoniche cubiche ridotte', np.linalg.det(np.dot(Phi_vP, Phi_vP.T)))
        mNa.append( mN[0])
        SNs += SN[0, 0] * np.exp(log_ev / log_evidence_vP[0])
        if ifprintmodpred: print('predizione a k=0 con polinomi di grado massimo', 2 * M_v, ':', mN[0], '\n')

    sr = np.var(np.array(mNa)) / M_tot ** 2 + SNs / (sum(np.exp(np.array(log_evidence_vP) / log_evidence_vP[0]))) ** 2

    mr = mNs / sum(np.exp(log_evidence_vP / log_evidence_vP[0]))
    if ifprintfinal: print('model average prediction',mr, np.sqrt(sr))
    return mr, sr, log_evidence_vP


def normalize(v, axi):
    norm = np.linalg.norm(v, axis=int(axi))
    if norm.any == 0:
        return v
    else:
        return (v.T / norm).T


def computephicubicL(x, betha0, M_v, nL=0):
    Phi_vP = [np.ones((x.shape[1])) * np.sqrt(betha0)]
    contanumpol = 1
    PhiL = np.zeros((3, x.shape[1]))
    PhiL[0] = 1/np.sqrt(4*np.pi)*np.ones(x.shape[1])
    #PhiL[1] = np.sqrt(5/16/np.pi)*(3 * x[2] ** 2 - (np.linalg.norm(x, axis=0)) ** 2)/(np.linalg.norm(x, axis=0)) ** 2
    PhiL[1] = np.sqrt(9/256/np.pi)*(35 * (x[0] ** 4 + x[1] ** 4 + x[2] ** 4) - 27 * (np.linalg.norm(x, axis=0)) ** 4)/(np.linalg.norm(x, axis=0)) ** 4
    for i in range(1, M_v):

        s = cubicharmonics.computecubicar(i, x.T, True)
        #divido le armoniche cubiche s per r**2i cosi' da avere solo i contributi delle armoniche sferiche e normalizzo, ottengo le nuove armoniche srid
        sridnorm = s.T / (np.linalg.norm(x, axis=0)) ** (2 * i)
        srid = copy.copy(sridnorm.T)
        t = s.T
        s = copy.copy(t.T)
        contanumpol += min(s.shape[1], nL+1)
        test = np.zeros((s.shape[1], nL+1))
        # In PhiL ci sono le armoniche sferiche normalizzate (in \theta e \phi) (numero di armoniche sferiche: #L),
        # sto calcolando i prodotti scalari con srid ((#L,#x)(#x, #c)).T-->(#c, #L)
        PhicoeffL = (PhiL[:nL+1]@srid).T
        for j in range(nL+1):
            for q in range(s.shape[1]):
                test[q, j]=np.sum(srid[:,q]*PhiL[j,:])
        for j in range(PhicoeffL.shape[1]):
            pass
            #print(PhicoeffL[:,j], j)
        # Contraggo le armoniche cubiche ridotte di grado 2*M_v per la matrice dei coefficienti (#x, #c)(#c, #L)-->(#x, #L)
        Phirid = (PhiL[:nL+1]*(np.linalg.norm(x, axis=0)) ** (2 * i)).T#((srid@PhicoeffL).T*(np.linalg.norm(x, axis=0)) ** (2 * i)).T
        for j in range(min(s.shape[1], nL+1)):
            Phi_vP = np.append(Phi_vP, [Phirid[:, j] * np.sqrt(betha0)], axis=0)

        Phi_vP = np.array(Phi_vP)
    # contanumpol e' il numero di armoniche cubiche associate a M_v
    if M_v == 1: Phi_vP = np.array(Phi_vP)

    return Phi_vP, contanumpol

def computephicubichandL(x, betha0, M_v, nL=0):
    Phi_vP = [np.ones((x.shape[1])) * np.sqrt(betha0)]
    contanumpol = 1
    PhiL = np.zeros((5, x.shape[1]))
    PhiL[0] = 1/np.sqrt(4*np.pi)*np.ones(x.shape[1])
    PhiL[1] = np.sqrt(9/256/np.pi)*(35 * (x[0] ** 4 + x[1] ** 4 + x[2] ** 4) \
                                    - 27 * (np.linalg.norm(x, axis=0)) ** 4)/(np.linalg.norm(x, axis=0)) ** 4
    PhiL[2] = 1./32 * np.sqrt(13./np.pi)*(231*(x[0] ** 6 + x[1] ** 6 + x[2] ** 6) \
                                          - 315*(x[0] ** 4 + x[1] ** 4 + x[2] ** 4) * (np.linalg.norm(x, axis=0)) ** 2 \
                                          + 100*(np.linalg.norm(x, axis=0)) ** 6)/(np.linalg.norm(x, axis=0)) ** 6
    PhiL[3] = 1./256 * np.sqrt(17./np.pi)*(6435*(x[0] ** 8 + x[1] ** 8 + x[2] ** 8) \
                                          - 12012*(x[0] ** 6 + x[1] ** 6 + x[2] ** 6) * (np.linalg.norm(x, axis=0)) ** 2 \
                                          + 6930*(x[0] ** 4 + x[1] ** 4 + x[2] ** 4) * (np.linalg.norm(x, axis=0)) ** 4 \
                                          - 1225*(np.linalg.norm(x, axis=0)) ** 8)/(np.linalg.norm(x, axis=0)) ** 8
    PhiL[4] = 1. / 256 * np.sqrt(17. / np.pi) * (46189 * (x[0] ** 10 + x[1] ** 10 + x[2] ** 10) \
                                                 - 109395 * (x[0] ** 8 + x[1] ** 8 + x[2] ** 8) * (np.linalg.norm(x, axis=0)) ** 2 \
                                                 + 90090 * (x[0] ** 6 + x[1] ** 6 + x[2] ** 6) * (np.linalg.norm(x, axis=0)) ** 4 \
                                                 - 30030 * (x[0] ** 4 + x[1] ** 4 + x[2] ** 4) * (np.linalg.norm(x, axis=0)) ** 6 \
                                                 + 3402 * (np.linalg.norm(x, axis=0)) ** 10) / (np.linalg.norm(x, axis=0)) ** 10
    for i in range(1, M_v):

        contanumpol += nL+1

        Phirid = (PhiL[:nL+1]*(np.linalg.norm(x, axis=0)) ** (2 * i)).T
        for j in range( nL+1):
            Phi_vP = np.append(Phi_vP, [Phirid[:, j] * np.sqrt(betha0)], axis=0)

        Phi_vP = np.array(Phi_vP)
    # contanumpol e' il numero di armoniche cubiche associate a M_v
    if M_v == 1: Phi_vP = np.array(Phi_vP)

    return Phi_vP, contanumpol

def computephicubicsfL(x, betha0, M_v, nL=0):
    Phi_vP = [np.ones((x.shape[1])) * np.sqrt(betha0)]
    contanumpol = 1
    theta = np.linspace(0, np.pi, 1000)
    phi = np.linspace(0, 2*np.pi, 1000)
    ang = np.array([theta, phi]).T
    PhiL = np.zeros((3, theta.shape[0]))
    PhiL[0] = 1 / np.sqrt(4 * np.pi) * np.ones(theta.shape[0])
    PhiL[1] = np.sqrt(5 / 16 / np.pi) * (3 * np.cos(theta)**2 - 1) * np.sin(theta)
    PhiL[2] = np.sqrt(9 / 256 / np.pi) * (35 * np.cos(theta)**4 - 30 * np.cos(theta)**2 + 3) * np.sin(theta)
    for i in range(1, M_v):

        s = cubicharmonics.computecubicar(i, x.T, True)
        ssf = cubicharmonics.computecubicarangular(i, ang, True)
        #divido le armoniche cubiche s per r**2i cosi' da avere solo i contributi delle armoniche sferiche e normalizzo, ottengo le nuove armoniche srid
        contanumpol += min(s.shape[1], nL+1)
        test = np.zeros((s.shape[1], nL+1))
        # In PhiL ci sono le armoniche sferiche normalizzate (in \theta e \phi) (numero di armoniche sferiche: #L),
        # sto calcolando i prodotti scalari con srid ((#L,#x)(#x, #c)).T-->(#c, #L)
        PhicoeffL = (PhiL[:nL+1]@ssf).T * np.pi/1000*2*np.pi/1000
        for j in range(nL+1):
            for q in range(s.shape[1]):
                test[q, j]=np.sum(ssf[:,q]*PhiL[j,:])* np.pi/1000*2*np.pi/1000
                print(test[q, j],q, j)
        for j in range(PhicoeffL.shape[1]):
            print(PhicoeffL[:,j], j)
        # Contraggo le armoniche cubiche ridotte di grado 2*M_v per la matrice dei coefficienti (#x, #c)(#c, #L)-->(#x, #L)
        Phirid = (s@PhicoeffL)
        for j in range(min(s.shape[1], nL+1)):
            Phi_vP = np.append(Phi_vP, [Phirid[:, j] * np.sqrt(betha0)], axis=0)

        Phi_vP = np.array(Phi_vP)
    # contanumpol e' il numero di armoniche cubiche associate a M_v
    if M_v == 1:Phi_vP = np.array(Phi_vP)

    return Phi_vP, contanumpol


def computephicubic(x, betha0, M_v):
    contanumpol = 1
    Phi = [np.ones((x.shape[1])) * np.sqrt(betha0)]
    for i in range(1, M_v):
        s = cubicharmonics.computecubicar(i, x.T, False)

        contanumpol += s.shape[1]

        for j in range(s.shape[1]):
            Phi = np.append(Phi, [s[:, j] * np.sqrt(betha0)], axis=0)

        Phi = np.array(Phi)

    # contanumpol e' il numero di armoniche cubiche associate a M_v
    if M_v == 1: Phi = np.array(Phi)
    return Phi, contanumpol


def cubicarray(MM, pr=False):
    if pr: print('numero di elementi: ', len(MM))
    # In input bisogna fornire la lista dei monomi M. Ogni elemento della lista M e'
    # una lista [a,b,c] tale per cui M_i=x^{2a}y^{2b}z^{2c} e a+b+c=n, dove n e' il grado del monomio.
    # I monomi in input comprendono tutte le permutazioni.
    # Voglio ora applicare le permutazioni x,y,z in modo da simmetrizzare i monomi e ottenere le armoniche cubiche.
    lenM = len(MM)
    M = []
    # Converto M a una lista di liste
    for i in range(len(MM)):
        M.append(list(MM[i]))
    Mt = copy.copy(M)
    # devo prima capire nella base dei monomi quali sono legati da permutazione. posso nominarli.
    nameC = []
    conta = 0
    contaci = 0
    if pr: print('numero cicli teornicamente necessari', int(len(M) ** 2 * 3 / 2))
    for i in range(lenM):
        rep = M[i]
        # rep (M[i]) e' il mio rappresentante nella lista,
        # voglio vedere quali altri elementi sono equivalenti per permutazione a rep
        nameC_i = []
        if not (rep in Mt): continue
        # se rep non sta nella lista temporanea Mt significa che e' equivalente a un elemento precedente: salto il ciclo
        nameC_i.append(np.array(rep))
        # in nameC_i inserisco gli elementi equivalenti per simmetria a rep
        for j in range(i + 1, lenM):
            # parto a cercare dall'elemento successivo nella lista, gli altri li ho gia' sicuramente controllati.
            Mjt = list(copy.copy(M[j]))

            if rep[0] in Mjt:
                Mjt.remove(rep[0])
                if rep[1] in Mjt:
                    Mjt.remove(rep[1])
                    if rep[2] in Mjt:
                        # rep e M[j] sono simmetrici per permutazione!
                        nameC_i.append(np.array(M[j]))
                        Mt.remove(M[j])

                        # rimuovo da Mt l'elemento M[j] che e' simmetrico per permutazione a rep
                        conta += 1
        contaci += len(nameC_i)
        nameC.append(nameC_i)
        # aggiungo la lista degli elementi associati a M[i], nameC_i, alla lista degli elementi NON equivalenti, nameC!
    if contaci != len(M): raise ValueError
    if pr: print('numero cicli compiuti', conta)
    if pr: print('numero di elementi non equivalenti per simmetria cubica: ', len(nameC))
    # in output c'e' la lista degli elementi che formano una classe di equivalenza.
    return nameC


def datadicG(data, sigmadata, G):
    dicdata = {}

    dicsigma = {}

    for i in range(len(G)):
        dicdata[tuple(G[i])] = data[i]
        dicsigma[tuple(G[i])] = sigmadata[i]

    return dicdata, dicsigma


def datiplot(C, dic, dics):
    gplot = []
    dataplot = []
    datasigmaplot = []
    for i in range(len(C)):
        gplot.append(C[i][0])
        dataav = 0
        datasigma = 0
        for j in range(len(C[i])):
            dataav += dic[tuple(C[i][j])] / len(C[i])
            datasigma += (dics[tuple(C[i][j])] / len(C[i])) ** 2
        dataplot.append(dataav)
        datasigmaplot.append(np.sqrt(datasigma))

    return np.array(gplot), np.array(dataplot), np.array(datasigmaplot)
