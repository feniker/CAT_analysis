from math import pi
from scipy.special import hyp2f1
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Im, Ith, Itr, Igun = 1.85e3, 1.85e3, 1.85e3, 3e3
coords_probes_1 = [0.2, -1.322]
coords_probes_2 = [0.2, -0.707]
coords_cathode = [0.065, -2.517]
mu0 = 4*pi*1e-7
B0 = mu0 #4*pi*I/a/c
psi0 = mu0*pi


def P(r, z, a):
    return 4*(a**2+r**2+z**2)**2*(r**2-2*(a**2+z**2))*\
        hyp2f1(3/4, 5/4, 2, 4*a**2*r**2/(a**2+r**2+z**2)**2)

def Q(r, z, a):
    return 15*a**2*r**2*(a**2-r**2+z**2)*\
        hyp2f1(7/4, 9/4, 3, 4*a**2*r**2/(a**2+r**2+z**2)**2)

def bz(r, z, a, I):
    return -(B0*I/a)*a**3*(P(r,z,a) - Q(r,z,a))/16/(a**2+r**2+z**2)**4.5

def br(r, z, a, I):
    return (B0*I/a)*3*a**3*z*r*(\
        5*a**2*r**2*hyp2f1(7/4,9/4,3,4*a**2*r**2/(a**2+r**2+z**2)**2)\
        + 2*(a**2+r**2+z**2)**2*hyp2f1(3/4,5/4,2,4*a**2*r**2/(a**2+r**2+z**2)**2))\
        /8/(a**2+r**2+z**2)**4.5
# [z0, a, I]
def b(r, z, loops):
    r, z = np.array(r), np.array(z)
    # loops[:,1] = loops[:,1]/a
    full_bz, full_br = 0, 0
    for loop in loops:
        full_bz+=bz(r.reshape(-1,1), z - loop[0], loop[1], loop[2])
        full_br+=br(r.reshape(-1,1), z - loop[0], loop[1], loop[2])
    return [full_bz, full_br]

def psi_once(r, z, a, I):
    return psi0*I*a/4*2*r**2*a/(r**2+a**2+z**2)**1.5*\
        hyp2f1(5/4,3/4,2,(2*r*a/(r**2+a**2+z**2))**2)
# [z0, a, I]
def psi(r, z, loops):
    r, z = np.array(r), np.array(z)
    # loops[:,1] = loops[:,1]/a
    full_psi = 0
    for loop in loops:
        full_psi+=psi_once(r.reshape(-1,1), z - loop[0], loop[1], loop[2])
    return full_psi

def get_coil(I,z1,r1,z2,r2,Nz,Nr):
    Idivided = np.ones(Nz*Nr)*I/Nz/Nr
    Zdivided = np.tile(np.linspace(z1,z2,Nz), Nr)
    Rdivided = np.repeat(np.linspace(r1,r2,Nr),Nz)
    return np.column_stack((Zdivided, Rdivided, Idivided))

def probe1Scale(x_):
    return -x_+16.4

def invprobe1Scale(x_):
    return 16.4-x_

def probe2Scale(x_):
    return x_-1.2

def invprobe2Scale(x_):
    return x_+1.2

def fromProcCm(r1, r2=None):
    if r2 is None: r2 = r1
    r = np.linspace(0,0.3,100)
    psi_center = psi(r, 0, loops).reshape(-1)
    inv_psi_center = CubicSpline(psi_center, r)
    normed_cathode = inv_psi_center(psi(coords_cathode[0], coords_cathode[1], loops).reshape(-1)[0])
    psi_probe1 = psi(r, coords_probes_1[1], loops).reshape(-1)
    inv_psi_probe1 = CubicSpline(psi_probe1, r)
    normed_coord1 = inv_psi_probe1(psi(r1/100*normed_cathode, 0, loops)).reshape(-1)[0]
    psi_probe2 = psi(r, coords_probes_2[1], loops).reshape(-1)
    inv_psi_probe2 = CubicSpline(psi_probe2, r)
    normed_coord2 = inv_psi_probe2(psi(r2/100*normed_cathode, 0, loops)).reshape(-1)[0]
    return normed_coord1, normed_coord2

def getCenter(r_, z_):
    r = np.linspace(0,0.4,100)
    psi_center = psi(r, 0, loops).reshape(-1)
    inv_psi_center = CubicSpline(psi_center, r)
    return np.array([inv_psi_center(psi(i, j, loops).reshape(-1))[0] for i,j in zip(r_,z_)])

def getProcCenter(r_,z_):
    return 100*getCenter(r_,z_)/getCenter([coords_cathode[0]],[coords_cathode[1]])

def set_loops(Im, Ith, Itr, Igun):
    global loops
    loops = np.vstack((get_coil(Im*80,0.26,0.20,0.34,0.30,2,6),
                get_coil(Im*80,-0.34,0.20,-0.26,0.30,2,6),
                get_coil(Ith*140,-1.04,0.15,-0.96,0.325,2,10),
                get_coil(Itr*32,-1.69,0.30,-1.61,0.34,2,5),
                get_coil(Itr*32,-2.14,0.30,-2.06,0.34,2,5),
                get_coil(Igun*62,-2.8075,0.134,-2.3925,0.224,20,2)))

loops = np.vstack((get_coil(Im*80,0.26,0.20,0.34,0.30,2,6),
                get_coil(Im*80,-0.34,0.20,-0.26,0.30,2,6),
                get_coil(Ith*140,-1.04,0.15,-0.96,0.325,2,10),
                get_coil(Itr*32,-1.69,0.30,-1.61,0.34,2,5),
                get_coil(Itr*32,-2.14,0.30,-2.06,0.34,2,5),
                get_coil(Igun*62,-2.8075,0.134,-2.3925,0.224,20,2)))

if __name__=="__main__":
    r = np.linspace(0,0.4,100)
    psi_center = psi(r, 0, loops).reshape(-1)
    inv_psi_center = CubicSpline(psi_center, r)
    interpolated_r = inv_psi_center(psi_center)

    """fig_center, ax_center = plt.subplots()
    fig_center.canvas.manager.set_window_title("Magnetic flux at the central plane") # or fig.canvas.set_window_title("Test")
    ax_center.plot(r, psi_center)
    ax_center.plot(interpolated_r, psi_center)
    ax_center.set(xlabel="r,m", ylabel="$\psi$,Wb")
    """
    # fig_center.show()
    # if coord triple probe r0 = 0.20, z0 = -1.2
    print("From proc to center cords: ", fromProcCm(getProcCenter([coords_probes_1[0]],[coords_probes_1[1]])[0],
    getProcCenter([coords_probes_2[0]],[coords_probes_2[1]])[0]), ' cm')
    print("Normalized radius of 2st Probe: ", 100*getCenter([coords_probes_1[0], coords_probes_2[0], coords_cathode[0]], 
    [coords_probes_1[1], coords_probes_2[1], coords_cathode[1]]), ' cm')

    
    # z = np.linspace(-3,2, 500)
    # r = np.linspace(0,0.6,100)
    # levels = np.sort(np.hstack((psi(np.linspace(0,0.6,12), 0.8,loops).reshape(-1), \
    #     psi(np.linspace(0.1,0.3,8), -1,loops).reshape(-1))))# np.linspace(1e-7,8e-5,15)
    
    # fig, ax = plt.subplots()
    # fig.canvas.manager.set_window_title("Magnetic lines in the CAT") # or fig.canvas.set_window_title("Test")
    # ax.contour(z, r, psi(r, z, loops), levels=levels, linewidths=0.5, zorder=0)
    # ax.add_patch(patches.Rectangle((0.26,0.20), 0.08, 0.1, edgecolor='black', fill=False))
    # ax.add_patch(patches.Rectangle((-0.34,0.20), 0.08, 0.1, edgecolor='black', fill=False))
    # ax.add_patch(patches.Rectangle((-1.04,0.15), 0.08, 0.175, edgecolor='black', fill=False))
    # ax.add_patch(patches.Rectangle((-1.69,0.30), 0.08, 0.04, edgecolor='black', fill=False))
    # ax.add_patch(patches.Rectangle((-2.14,0.30), 0.08, 0.04, edgecolor='black', fill=False))
    # ax.add_patch(patches.Rectangle((-2.8075,0.134), 0.415, 0.009, edgecolor='black', fill=False))
    # ax.set(xlabel="z,m", ylabel="r,m")
    # plt.show()