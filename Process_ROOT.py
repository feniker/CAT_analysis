import numpy as np
from scipy import signal
from Get_ROOT import get_shot

# Constants and initial tup
# Experiments[numShot, rawTe, rawne, filtTe, filtne]
# Experiments = {}
# time = []

NumCells = 14

tfft = 1.5e-3#1e-3 #s for plots
num = 2000  # number of separation to calculate std
Freq_cut = 10000 # for filter butterworth
R = 1  # Ohm resistance from which the current was calculated
POTfactor = 200  # V coefficient of division temper
Ufactor = 10  # V coefficient of division temper
Ifactor = 1 / R  # A coefficient of division zontok
W = 400 * 10 ** -4  # sm thickness of probe
L = 0.3  # sm length of probe
e = 4.8 * 10 ** -10  # statKulon elementary charge
Mp = 1.67 * 10 ** -24  # gram mass of a proton
S = np.pi * W * L  # surface of the cylindrical probe

butter = True # 1 - butterworth; 0 - aver

koef = {"TeProbeMirror": Ufactor, "neProbeMirror": Ifactor, "PotentialProbeMirror": POTfactor,
        "TeProbePG": Ufactor, "neProbePG": Ifactor, "PotentialProbePG": POTfactor,
        "TeProbeCenter": Ufactor, "neProbeCenter": Ifactor, "PotentialProbeCenter": POTfactor,
        "CathodePG": 100., "AnodePG": 100., "MainSolenoid1V_kA": -1.e3, "ThermoSolenoid1V_kA": -1.e3,
        "GunSolenoid2V_kA": -1.e3/2., "GanArc2": 2.e3/0.333/1., "GunArc": 3.e3/0.333/4.2, 
        "VacPG1": -1./100., "VacPB4": -1./100., "PotentialCathode": 100., "Diamag": 3./25.,
        'MirnovProbe1': 1., 'MirnovProbe2': 1., 'MirnovProbe3': 1., 'MirnovProbe4': 1., 
        'MirnovProbe5': 1., 'MirnovProbe6': 1., 'MirnovProbe7': 1., 'MirnovProbe8': 1.,
        'MirnovProbe9': 1., 'MirnovProbe10': 1., 'MirnovProbe11': 1., 'MirnovProbe12': 1.}


def fromEvToErg(Te):
    return Te * 1.6 * 10 ** -12

def fromAmpToStatAmp(I):
    return I * 3 * 10 ** 9

def temper(u23):
    Te = Ufactor * u23 / np.log(2)  # eV electron temperature
    return Te

def elDen(I, Te):
    # cutting off all temperature less than 1/30 of max
    Te = np.abs(Te)
    epsilon = np.max(Te[Te.nonzero()])/30
    Te[Te < epsilon] = epsilon
    Te = fromEvToErg(Te)
    ne = Ifactor * fromAmpToStatAmp(I) * np.exp(0.5) / S / e * np.sqrt(2 * Mp / Te)  # sm-3 electron density
    return ne

def timeToInd(t, tstart, tfinish, N):
    if tstart < tfinish:
        return int(N*(t-tstart)/(tfinish - tstart))
    else:
        return 0

def freqToInd(f, fmin, fmax, N):
    if fmin < fmax:
        return int(N*(f-fmin)/(fmax - fmin))
    else:
        return 0

def getMean(ar, t1 = 0, t2 = 4e-3):
    N1 = timeToInd(t1, ar[0][0], ar[0][-1], ar[0].size)
    N2 = timeToInd(t2, ar[0][0], ar[0][-1], ar[0].size)
    return np.mean(ar[1][N1:N2])

def but_filt(curv, dt, fc=4000):
    # fc = 0.006  # Cut-off frequency of the filter
    w = fc*dt  # Normalize the frequency
    #sos = signal.butter(2, fc, 'lp', fs = 1/dt, output='sos')
    b, a = signal.butter(2, w, 'lowpass', output='ba')
    ans = signal.filtfilt(b, a, curv)
    ans = signal.filtfilt(b, a, ans[::-1])
    return ans[::-1]

def aver(interval, num=100):
    window= np.ones(int(num))/float(num)
    return np.convolve(interval, window, 'same')

def disp(a, num=300):
    buf = (aver(a) - a)*(aver(a) - a)
    return np.sqrt(aver(buf, num = num))

def transform(U, Uname):
    try:
        return koef[Uname]*U
    except KeyError:
        print("CAN'T TRANSFORM, Process_ROOT.transform()")
        return U

def getUca(Archive_name, names = {"Vacuum": ["CathodePG", "AnodePG"]}, shots = [0]):
    Exp = {}
    dataset = get_shot(Archive_name, names, shots)
    for kust in dataset.keys():
        for tup in dataset[kust]:
            numshot = int(tup[5])
            if not Exp.get(numshot) is None:
                if tup[4] == "CathodePG": Exp[numshot][0] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
                if tup[4] == "AnodePG": Exp[numshot][1] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            else:
                Exp[numshot] = [0]*2
    for i in Exp.keys():
        Exp[i] = [Exp[i][0][0], 100*Exp[i][1][1] - 100*Exp[i][0][1]]
    return Exp

def getLeafs(Archive_name, names = {"Vacuum": ["CathodePG", "AnodePG"]}, shots = [0]):
    Exp = {}
    dataset = get_shot(Archive_name, names, shots)
    names_flatten = list(np.concatenate(list(names.values())))
    for kust in dataset.keys():
        for tup in dataset[kust]:
            numshot = int(tup[5])
            if not Exp.get(numshot):                  Exp[numshot] = [0]*len(names_flatten)
            if tup[4] in names_flatten:             
                # print(names_flatten.index(tup[4]), tup[4])
                Exp[numshot][names_flatten.index(tup[4])] = [np.linspace(tup[1], tup[2], tup[3]), transform(tup[0].copy(), tup[4])]
    return Exp


def getTriple(Archive_name, names, shots):
    # Return dict with the following structure: {'numshot': [(time, TeMir), (time, neMir), (time, potMir),
    # (time, TePG), (time, nePG), (time, potPG)}
    # But you shoul remember that all data is raw and it's need to be translated to physical values 
    # Every array has own time array
    # Reading file
    NumCells = 10
    Exp = {}
    dataset = get_shot(Archive_name, names, shots)
    for kust in dataset.keys():
        for tup in dataset[kust]:
            numshot = int(tup[5])
            if not numshot in Exp:                  Exp[numshot] = [0]*NumCells
            if tup[4] == "TeProbeMirror":           Exp[numshot][0] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "neProbeMirror":           Exp[numshot][1] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "PotentialProbeMirror":    Exp[numshot][2] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "TeProbePG":               Exp[numshot][3] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "neProbePG":               Exp[numshot][4] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "PotentialProbePG":        Exp[numshot][5] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
    return Exp

def transformTriple(Exp):
    for i in Exp.keys():
        # temperature isn't absolutely here
        Exp[i][0][1] = temper(Exp[i][0][1] - np.mean(Exp[i][0][1][-2000:]))
        Exp[i][1][1] = -1*elDen(Exp[i][1][1], Exp[i][0][1])
        Exp[i][1][1] = Exp[i][1][1] - np.mean(Exp[i][1][1][-2000:]) # subtracting background
        Exp[i][2][1] = Exp[i][2][1]*POTfactor # transformation of potential
        # same procedure with PG probe
        # temperature isn't absolutely here
        Exp[i][3][1] = temper(Exp[i][3][1] - np.mean(Exp[i][4][1][-2000:]))
        Exp[i][4][1] = -1*elDen(Exp[i][4][1], Exp[i][3][1])
        Exp[i][4][1] = Exp[i][4][1] - np.mean(Exp[i][4][1][-2000:]) # subtracting background
        Exp[i][5][1] = Exp[i][5][1]*POTfactor # transformation of potential
        
        # filter data with Butterworth filter (two sides in order to avoid a phase raid)
        dt = (Exp[i][0][0][-1] - Exp[i][0][0][0])/len(Exp[i][0][0])
        Exp[i][6] = [Exp[i][0][0], but_filt(Exp[i][0][1], dt, fc=Freq_cut)]
        Exp[i][7] = [Exp[i][1][0], but_filt(Exp[i][1][1], dt, fc=Freq_cut)]
        Exp[i][8] = [Exp[i][3][0], but_filt(Exp[i][3][1], dt, fc=Freq_cut)]
        Exp[i][9] = [Exp[i][4][0], but_filt(Exp[i][4][1], dt, fc=Freq_cut)]
    return Exp

def getMagnetic(Archive_name, names, shots):
    # Return dict with the following structure: {'numshot': [(time, CurMainSol), (time, CurThermoSol), (time, GunArc),
    # (time, GunArc2), (time, GunSol), (time, CathodePG), (time, AnodePG), (time, VacPG1), (time, VacPB4), (time, PotCentralDisk)}
    # But you shoul remember that all data is raw and it's need to be translated to physical values 
    # Every array has own time array
    # Reading file
    Exp = {}
    dataset = get_shot(Archive_name, names, shots)
    for kust in dataset.keys():
        for tup in dataset[kust]:
            numshot = int(tup[5])
            if not numshot in Exp:              Exp[numshot] = [0]*NumCells
            if tup[4] == "MainSolenoid1V_kA":   Exp[numshot][0] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "ThermoSolenoid1V_kA": Exp[numshot][1] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "GunArc":              Exp[numshot][2] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "GanArc2":             Exp[numshot][3] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "GunSolenoid2V_kA":    Exp[numshot][4] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "AnodePG":           Exp[numshot][5] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "CathodePG":             Exp[numshot][6] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "VacPG1":              Exp[numshot][7] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "VacPB4":              Exp[numshot][8] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
            if tup[4] == "PotentialCathode":    Exp[numshot][9] = [np.linspace(tup[1], tup[2], tup[3]), tup[0].copy()]
    return Exp

def transformMagnetic(Exp):
    for i in Exp.keys():
        Exp[i][0][1] = -Exp[i][0][1]
        Exp[i][1][1] = -Exp[i][1][1]
        Exp[i][2][1] = 3./0.333/4.2*Exp[i][2][1]
        Exp[i][3][1] = 2./0.333/1.*Exp[i][3][1]
        Exp[i][4][1] = -1./2.*Exp[i][4][1]
        Exp[i][5][1] = 100*Exp[i][5][1] # Cathode and Anode rearranged
        Exp[i][6][1] = 100*Exp[i][6][1]
        Exp[i][5][1][0], Exp[i][6][1][0] = 0, 0
        Exp[i][7][1] = -1./100.*Exp[i][7][1]
        Exp[i][8][1] = -1./100.*Exp[i][8][1]
        Exp[i][9][1] = 100.*Exp[i][9][1]
    return Exp

if __name__ == "__main__":
    ArchiveName = "/home/murakhtin/test.root"
    names = {"Vacuum": ["TeProbeMirror", "neProbeMirror", "PotentialProbeMirror",
                    "TeProbePG", "neProbePG", "PotentialProbePG"]}
    shots = [0] # 4 shots is max
    data = getTriple(ArchiveName, names, shots)
    data = transformTriple(data)
    print(data)