import numpy as np
import copy


# Calcolo le arominche cubiche, le operazioni di simmetria sono le permutazioni di x,y,z. Le operazioni sarebbero 48 ma se uso solo monomi di ordine pari ho gia' considerato tutte le rilessioni.
# Definisco $M_i$ i monomi del grado appropriato, allora per ottenere le armoniche cubiche ,$C_i$:
# $C_i=\frac{1}{48}\sum_{R}RM_i$
#
# I monomi di grado minore (2) sono $x^2, y^2, z^2$. ovviamente devo ottenere $x^2+y^2+z^2$.

def Gvecgenerateall(nk):
    G = np.zeros((nk, 3))
    conta = 0
    G[0] = np.array([0, 0, 0])
    nkp = int(np.power(nk, 1 / 3)) + 1
    for i in range(0, nkp):
        for j in range(0, nkp):
            for k in range(0, nkp):
                if i == 0 and j == 0 and k == 0: continue
                conta += 1
                if conta == nk:
                    return G
                G[conta] = np.array([i, j, k])

    return G


def basis(n):
    # n e' il grado del monomio,
    # dato che uso solo monomi di ordine pari voglio sapere quanti x^2, y^2, z^2 ci sono in ogni monomio.
    # ogni monomio e' una lista [a,b,c] tale per cui M_i=x^{2a}y^{2b}z^{2c} e a+b+c=n.
    M = []
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            for k in range(0, n + 1):
                if (i + j + k == n):
                    M_i = [2 * i, 2 * j, 2 * k]
                    M.append(M_i)
    return M


def cubicarmonic(M, pr=False):
    if pr: print('numero di monomi nella base: ', len(M))
    # In input bisogna fornire la lista dei monomi M. Ogni elemento della lista M e'
    # una lista [a,b,c] tale per cui M_i=x^{2a}y^{2b}z^{2c} e a+b+c=n, dove n e' il grado del monomio.
    # I monomi in input comprendono tutte le permutazioni.
    # Voglio ora applicare le permutazioni x,y,z in modo da simmetrizzare i monomi e ottenere le armoniche cubiche.
    lenM = len(M)
    Mt = copy.copy(M)
    # devo prima capire nella base dei monomi quali sono legati da permutazione. posso nominarli.
    nameC = []
    conta = 0
    contaci = 0
    if pr: print('numero cicli teornicamente necessari', int(len(M) ** 2 * 3 / 2))
    for i in range(lenM):
        rep = M[i]
        # rep (M[i]) e' il mio rappresentante nella lista dei monomi,
        # voglio vedere quali altri monomi sono equivalenti per permutazione a rep
        nameC_i = []
        if not (rep in Mt): continue
        # se rep non sta nella lista temporanea Mt significa che e' equivalente a un monomio precedente: salto il ciclo
        nameC_i.append(rep)
        # in nameC_i inserisco i monomi equivalenti per simmetria a rep
        # in modo da poter costruire l'armonica cubica C_i associata a M[i]
        for j in range(i + 1, lenM):
            # parto a cercare dall'elemento successivo nella lista, gli altri li ho gia' sicuramente controllati.
            Mjt = copy.copy(M[j])
            if rep[0] in Mjt:
                Mjt.remove(rep[0])
                if rep[1] in Mjt:
                    Mjt.remove(rep[1])
                    if rep[2] in Mjt:
                        # rep e M[j] sono simmetrici per permutazione!
                        nameC_i.append(M[j])
                        # aggiungo M[j] alla lista dei monomi equivalenti a rep
                        Mt.remove(M[j])
                        # rimuovo da Mt l'elemento M[j] che e' simmetrico per permutazione a rep
                        conta += 1
        contaci += len(nameC_i)
        nameC.append(nameC_i)
        # aggiungo la lista dei monomi associati a M[i], nameC_i, alla lista delle armoniche cubiche, nameC!
    if contaci != len(M): raise ValueError
    if pr: print('numero cicli compiuti', conta)
    if pr: print('numero di armoniche cubiche: ', len(nameC))
    # in output c'e' la lista dei monomi che formano una armonica sferica come definiti in basis.
    return nameC


def shape(lst):
    length = len(lst)
    shp = tuple(shape(sub) if isinstance(sub, list) else 0 for sub in lst)
    if any(x != 0 for x in shp):
        return length, shp
    else:
        return length


def computecubicar(N, G, pr=False):
    # G e' la lista di punti 3D dove sto samplando le armoniche cubiche
    C = cubicarmonic(basis(N))

    if pr: print('numero di armoniche cubiche di grado ', N * 2, ' : ', len(C))

    for i in range(len(C)):
        if pr: print(C[i])

    M = len(C)

    nk = len(G)

    Cfun = np.zeros((nk, M))
    for i in range(0, nk):
        for s in range(M):
            for t in range(len(C[s])):
                Cfun[i, s] += (G[i, 0] ** (np.array(C[s])[t][0])) * \
                                  (G[i, 1] ** (np.array(C[s])[t][1])) * (G[i, 2] ** (np.array(C[s])[t][2]))
    return Cfun

