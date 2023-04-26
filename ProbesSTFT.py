import warnings
warnings.filterwarnings("ignore")
import Process_ROOT as pr
import Magn as mg
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

ArchiveName = "/home/murakhtin/test.root"

def get_frot(U, shot, butter=True):
    names = {"Magnetic": ["MainSolenoid1V_kA", "ThermoSolenoid1V_kA"],
            "PlasmaGun": ["GunSolenoid2V_kA"]}
    data = pr.getLeafs(ArchiveName, names, [shot])
    Im = pr.getMean(data[shot][0])
    Ith = pr.getMean(data[shot][1])
    Itr = Ith
    Igun = pr.getMean(data[shot][2], 0, 2e-3)
    # Im, Ith, Itr, Igun = 1.85e3, 1.85e3, 1.85e3, 3.e3
    # print(Im, Ith, Itr, Igun)
    mg.set_loops(Im, Ith, Itr, Igun)
    B_ax = mg.b(0, -2.6, mg.loops)[0].flatten()[0]
    print(f"In shot {shot}","Bz = %.1f kGs" %(B_ax*10))
    dt = data[shot][0][0][1] - data[shot][0][0][0] # time interval for buterworth filter, 
    # it can be wrong because dt is taken from another time series (MainSolenoid)
    if butter:
        return pr.but_filt(U, dt, fc=1e4)/5.5/1/(B_ax)*10**4/2/np.pi   # 5.5 = r, 0.1 = dr, 3 = B
    else:
        return U/5.5/1/B_ax*10**-1/2/np.pi  # 5.5 = r, 1 = dr, 3 = B
    
if __name__=="__main__":
    shots = [0] # 4 shots is max
    names = {"Vacuum": ["PotentialProbeMirror","PotentialProbePG","PotentialProbeCenter", "CathodePG", "AnodePG"]}
    data = pr.getLeafs(ArchiveName, names, shots)
    Uca = {i: data[i][4][1] - data[i][3][1] for i in data.keys()}
    time_Uca = {i: data[i][4][0] for i in data.keys()}

    t_step, t_fft, t_start, t_finish = 0.03, 0.3, 0, 5 #ms
    for shot in sorted(data.keys()):
        # plt.plot(time_Uca[shot], Uca[shot])
        # plt.show()
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        fig.patch.set_facecolor('xkcd:mint green')
        list_labels = ['Cone', 'Gun', "Center"]
        frot = get_frot(Uca[shot], shot)/1000
        Zxx = []
        vmax = 0
        for i in [1,0,2]:
            time, sig = data[shot][i]
            dt = time[1] - time[0]                                                          # time step
            N_start = pr.timeToInd(t_start, time[0]*1000, time[-1]*1000, len(time))
            N_finish = pr.timeToInd(t_finish, time[0]*1000, time[-1]*1000, len(time))
            nperseg = round((N_finish - N_start) * t_fft / (t_finish - t_start))            # number of points to one segment
            noverlap = round(nperseg - (N_finish - N_start) * t_step / (t_finish - t_start))# number of points to one segment
            filted_signal = pr.but_filt(sig[N_start:N_finish], dt, fc = 3000)               # filted signal used for subtracting high background signal
            f, t, Zxx = signal.stft(sig[N_start:N_finish] - filted_signal, 1/dt, \
                nperseg=nperseg, noverlap=noverlap)
            Nbot, Nup = pr.freqToInd(10, f[0]/1000, f[-1]/1000, len(f)), pr.freqToInd(150, f[0]/1000, f[-1]/1000, len(f))
            ax_cor = ax[i]
            # ax_cor.pcolormesh(t*1000 - 1, f / 1000, np.log(np.abs(Zxx)), vmin=np.log(np.mean(np.abs(Zxx[Nbot:Nup,:]))*0.7),\
            #     vmax=np.log(np.max(np.abs(Zxx[Nbot:Nup,:]))), shading='gouraud', cmap = 'magma')
            if (i == 1):
                vmax = np.max(np.abs(Zxx[Nbot:Nup,:]))
            pc = ax_cor.pcolormesh(t*1000 + t_start, f / 1000, np.abs(Zxx), vmin=0,\
                vmax=vmax, shading='gouraud', cmap = 'magma')
            fig.colorbar(pc, ax=ax_cor)
            ax_cor.plot(time_Uca[shot]*1000, frot, ls='--', label = "1 mode")
            ax_cor.plot(time_Uca[shot]*1000, 0.5*frot, ls=':', label = "2 mode")
            # ax_cor.plot(time_Uca[shot]*1000, 2*frot(Uca[shot], shot)/1000)
            ax_cor.set(title=list_labels[i] + str(shot), ylabel='Frequency, kHz', xlabel='Time, ms', xlim=(t_start, t_finish), ylim=(0,150))
            ax_cor.legend()
        fig.suptitle("Spectrograms of shot: " + str(shot), size = 'xx-large')
        fig.show()
    plt.show()