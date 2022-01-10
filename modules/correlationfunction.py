import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft


def correlation(nk, nsnap, L, en_at, pos_at):

    q = np.zeros((nk, nsnap), dtype=np.complex_)
    q1 = np.zeros((nk, nsnap), dtype=np.complex_)
    for i in range(nk):
        q1[i] = np.sum(en_at[:, :] * np.exp(-1j * pos_at[:, :, 0] * 2 * i * np.pi / L), axis=1)
    qm = np.sum(q1, axis=1) / nsnap
    q = np.transpose(np.transpose(q1) - qm)

    corr = np.zeros((nk, nsnap), dtype=np.complex_)
    result = np.zeros(2 * nsnap, dtype=np.complex_)
    for j in range(nk):
        result = signal.correlate((q[j, :]), q[j, :], mode='full', method='fft')
        v = [result[i] / (len(q[j, :]) - abs(i - (len(q[j, :])) + 1)) for i in range(len(result))]
        corr[j] = np.array(v[int(result.size / 2):])

    ft = np.zeros((nk, int(nsnap / 2)), dtype=np.complex_)
    for i in range(nk):
        ft[i] = fft(np.real(corr[i][:int(nsnap / 2)]))

    fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
    plt.imshow(np.real(ft[:nk, :nk]), interpolation='none', aspect='auto')
    plt.colorbar()
    plt.title(r'$ S_{qq}({\bf k}, \omega)$')
    plt.ylabel(r'k in units of $\frac{2\pi}{L}$')
    plt.xlabel(r'$\omega$ in units of $\frac{2\pi}{T}$')
    plt.savefig('kwftautocorr.pdf')
    return ft
