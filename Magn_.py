from math import pi
from scipy.special import hyp2f1
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROBE_1 = [0.2, -1.322]
PROBE_2 = [0.2, -0.707]
CATHODE = [0.065, -2.517]
MU_0 = 4*pi*1e-7
B_0 = MU_0 #4*pi*I/a/c
PSI_0 = MU_0*pi
Im, Ith, Itr, Igun = 1.85e3, 1.85e3, 1.85e3, 3e3


def psiOnce(z, r, a, I):
    """
    Return Psi for one loop
    """
    return PSI_0*I*a/4*2*r**2*a/(r**2+a**2+z**2)**1.5*\
        hyp2f1(5/4,3/4,2,(2*r*a/(r**2+a**2+z**2))**2)

def psi(zPoints, rPoints, loops):
    """
    Return (Nr,Nz) array of psi value
    """
    zPoints, rPoints = np.array(zPoints), np.array(rPoints)
    fullPsi = 0
    for loop in loops:
        fullPsi+=psiOnce(zPoints - loop[0], rPoints.reshape(-1,1), loop[1], loop[2])
    return fullPsi

def getCoil(I,z1,r1,z2,r2,Nz,Nr):
    """
    Get set of thin string with total current I.
    Currents are represent as set of thin strings.
    output: [[z1, r1, I1], ... , [zN, rN, IN]]
    """
    currentString = np.ones(Nz*Nr)*I/Nz/Nr # Current have to be 
    zString = np.tile(np.linspace(z1,z2,Nz), Nr)
    rString = np.repeat(np.linspace(r1,r2,Nr),Nz) # iterate over all cases
    return np.column_stack((zString, rString, currentString))

def scaleToAbs(name, rCoord):
    if name == "Plasma Gun":
        return -rCoord+0.164
    elif name == "Cone":
        return rCoord-0.012
    elif name == "Center":
        return -rCoord+0.24
    return None

def scaleToLocal(name, rCoord):
    if name == "Plasma Gun":
        return -rCoord+0.164
    elif name == "Cone":
        return rCoord+0.012
    elif name == "Center":
        return -rCoord+0.24
    return None

def getLocalCoord(points, loops, isPercent = True):
    rAxis = np.linspace(0,0.3,100)
    localCoords = []
    if isPercent:
        normed_cathode = getCenterCoord([CATHODE], loops)[0]
        rKoef = normed_cathode/100
    for r, z in points:
        psiPoint = psi(z, rAxis, loops).ravel()
        invPsiPoint = CubicSpline(psiPoint, rAxis)
        localCoords.append(invPsiPoint(psi(0, r*rKoef, loops).ravel()[0]).ravel()[0])
    return np.array(localCoords)

def getCenterCoord(points, loops):
    rAxis = np.linspace(0,0.4,100)
    psiCenter = psi(0, rAxis, loops).ravel()
    invPsiCenter = CubicSpline(psiCenter, rAxis)
    return np.array([invPsiCenter(psi(z, r, loops).ravel()[0]) for r,z in points])

def getCenterPercent(points, loops):
    return 100*getCenterCoord(points, loops)/getCenterCoord([CATHODE], loops)

def getLoops(mirrorCur, thermoCur, transportCur, gunCur):
    # don't forget to mupltiply on turns number
    loops = np.vstack((getCoil(mirrorCur*80,0.26,0.20,0.34,0.30,3,6),
                getCoil(mirrorCur*80,-0.34,0.20,-0.26,0.30,3,6),
                getCoil(thermoCur*140,-1.04,0.15,-0.96,0.325,3,10),
                getCoil(transportCur*32,-1.69,0.30,-1.61,0.34,3,10),
                getCoil(transportCur*32,-2.14,0.30,-2.06,0.34,3,10),
                getCoil(gunCur*62,-2.8075,0.134,-2.3925,0.224,30,3)))
    return loops

if __name__=="__main__":
    loops = getLoops(Im, Ith, Itr, Igun)
    r = np.linspace(0,0.4,100)
    psi_center = psi(0, r, loops).ravel()
    inv_psi_center = CubicSpline(psi_center, r)
    interpolated_r = inv_psi_center(psi_center)

    psi_center = psi([0, PROBE_1[1], PROBE_2[1]], r, loops)
    fig_center, ax_center = plt.subplots()
    fig_center.canvas.manager.set_window_title("Magnetic flux at the central plane") # or fig.canvas.set_window_title("Test")
    ax_center.plot(r, psi_center[:,0], label = "Center")
    ax_center.plot(r, psi_center[:,1], label = "1 Probe")
    ax_center.plot(r, psi_center[:,2], label = "2 Probe")
    # ax_center.plot(interpolated_r, psi_center, label = "Inv Center")
    ax_center.set(xlabel="r,m", ylabel="$\psi$,Wb")
    ax_center.legend()
    fig_center.show()
    # if coord triple probe r0 = 0.20, z0 = -1.2
    Proc1 = 150
    points = [[Proc1, PROBE_1[1]], [Proc1, PROBE_2[1]]]
    print(getLocalCoord(points, loops)*1e3)
    print("Gun Probe to %.1f" %(scaleToLocal("Plasma Gun", getLocalCoord(points, loops)[0])*1e3))
    print("Mirror Probe to %.1f" %(scaleToLocal("Cone", getLocalCoord(points, loops)[1])*1e3))

    
    z = np.linspace(-3,2, 500)
    r = np.linspace(0,0.6,100)
    levels = np.sort(np.hstack((psi(0.8, np.linspace(0,0.6,12), loops).reshape(-1), \
        psi(-1, np.linspace(0.1,0.3,8), loops).reshape(-1))))# np.linspace(1e-7,8e-5,15)
    
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Magnetic lines in the CAT") # or fig.canvas.set_window_title("Test")
    ax.contour(z, r, psi(z, r, loops), levels=levels, linewidths=0.5, zorder=0)
    ax.add_patch(patches.Rectangle((0.26,0.20), 0.08, 0.1, edgecolor='black', fill=False))
    ax.add_patch(patches.Rectangle((-0.34,0.20), 0.08, 0.1, edgecolor='black', fill=False))
    ax.add_patch(patches.Rectangle((-1.04,0.15), 0.08, 0.175, edgecolor='black', fill=False))
    ax.add_patch(patches.Rectangle((-1.69,0.30), 0.08, 0.04, edgecolor='black', fill=False))
    ax.add_patch(patches.Rectangle((-2.14,0.30), 0.08, 0.04, edgecolor='black', fill=False))
    ax.add_patch(patches.Rectangle((-2.8075,0.134), 0.415, 0.009, edgecolor='black', fill=False))
    ax.scatter(loops[:,1], loops[:,0])
    ax.set(xlabel="z,m", ylabel="r,m")
    plt.show()