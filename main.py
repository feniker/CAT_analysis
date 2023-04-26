import numpy as np
import matplotlib.pyplot as plt
from math import pi

N = 24
N_t = 1000
A = .03
noise = 0.2
f = 10
time = np.linspace(0, 10, N_t).reshape((N_t,-1))
phi = np.linspace(0, 2*pi*(N-1)/N, N)
phi = np.reshape(phi, (1, -1))

M = 5 + np.sin(2*pi*f*time + 3*phi)
M += A*np.random.normal(0, noise, M.shape)

plt.plot(time, M[:,0])
plt.plot(time, M[:,5])
# plt.polar(phi.flatten(), M[0,:])
plt.show()

# print(U.shape)

U, S, V = np.linalg.svd(M)
print(S)
plt.plot(time, U[0,:])
plt.plot(time, U[1,:])
plt.show()
plt.plot(phi.flatten(), M[0,:])
plt.plot(phi.flatten(), -np.sqrt(S[0])*V[0,:])
plt.plot(phi.flatten(), np.sqrt(S[1])*V[1,:])
plt.plot(phi.flatten(), np.sqrt(S[2])*V[2,:])
plt.show()

for i in [0, 6, 9]:
    X = np.fft.rfft(M[:,0])
    Y = np.fft.rfft(M[:,i])
    freq = np.fft.rfftfreq(M[:,0].size, d=10/N_t)
    S01 = np.correlate(X, Y, mode = 'same')

    Theta = np.arctan(S01.imag/S01.real)
    Gamma = np.abs(S01)/np.sqrt(np.abs(np.correlate(X, X, mode = 'same')*np.correlate(Y, Y, mode = 'same')))

    # plt.plot(freq, Theta, label = str(i))
    plt.plot(freq, Gamma, label = str(i))
plt.legend()
plt.show()
