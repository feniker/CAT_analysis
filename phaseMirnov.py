from math import pi
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import Process_ROOT as pr

if __name__ == "__main__":
    ArchiveName = "/home/murakhtin/test.root"
    N = 12
    window_sz = 50
    shots = [4008] # 4 shots is max
    tStart, tFinish = 1/1e3, 3/1e3
    probeNames = ["MirnovProbe" + str(i+1) for i in range(N)]
    names = {"PlasmaDump": probeNames}
    data = pr.getLeafs(ArchiveName, names, shots)
    phi = np.linspace(0, 2*pi*(N-1)/N, N)
    for shot in data.keys():
        signs = []
        time = np.array(data[shot][0][0])
        newTime = time[(time>tStart) & (time<tFinish)]
        newTime = newTime[:len(newTime)//window_sz*window_sz].reshape(-1,window_sz).mean(1) 
        for sign in data[shot]:
            newSign = sign[1][(time>tStart) & (time<tFinish)]
            newSign = newSign[:len(newSign)//window_sz*window_sz].reshape(-1,window_sz).mean(1) 
            signs.append(newSign)
        signs = np.array(signs)

        fig, ax = plt.subplots(figsize=(17, 5))
        fig.patch.set_facecolor('xkcd:mint green')
        ax.pcolormesh(newTime*1000, range(1,13), signs)#, shading='gouraud')
        ax.vlines(x=[1,2,3],ymin=1,ymax=12,colors='b')
        fig.suptitle("Spectrograms of shot: " + str(shot), size = 'xx-large')
        fig.show()
        U, S, V = np.linalg.svd(signs.T)
        print(S/np.sum(S))
        fig2, ax2 = plt.subplots(figsize=(17, 5))
        ax2.plot(newTime, np.sqrt(S[0])*U[0,:])
        ax2.plot(newTime, np.sqrt(S[1])*U[1,:])
        fig2.show()
        fig3 = plt.figure(dpi=200)
        ax3 = fig3.add_subplot(projection='polar')
        # fig3, ax3 = plt.subplots(figsize=(17, 5), projection='polar')
        # plt.plot(phi.flatten(), signs.T[0,:])
        ax3.plot(phi.flatten(), -np.sqrt(S[0])*V[0,:])
        ax3.plot(phi.flatten(), np.sqrt(S[1])*V[1,:])
        ax3.plot(phi.flatten(), np.sqrt(S[2])*V[2,:])
        fig3.show()
        fig4, ax4 = plt.subplots(figsize=(17, 5))
        ax4.scatter(range(1,13), S/np.sum(S))#, shading='gouraud')
        ax4.set_ylim([0,1])
        fig4.show()
        plt.show()
    # np.fft.fft2()

        
    
    plt.show()