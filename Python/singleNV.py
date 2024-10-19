import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import seaborn as sns


def get_cartesian_vector(A, theta, phi):
    # returns the cartensian coordinates from it's spherical coordinates
    cart_vec = np.array([A * np.sin(theta) * np.cos(phi), A * np.sin(theta) * np.sin(phi), A * np.cos(theta)])

    return cart_vec

def get_spherical_vector(Avec):
    # returns the spherical coordinates from its cartesian coordinates
    A0 = np.sqrt(np.dot(Avec, Avec))
    theta = np.arccos(Avec[2] / A0)
    try:
        phi = np.arctan(Avec[1] / Avec[0])
    except ZeroDivisionError:
        phi = 0.
    if np.isnan(phi):
        phi = 0.
    if Avec[0] < 0:
        phi += np.pi

    return A0, theta, phi


# Constants 
# NV fine constants

D_0 = 2.87e9 # zero field splitting
D_E = 1.42e9 # zero field splitting for excited state

# Magnetic coupling constants (in SI units)
muB = 9.274e-24 # bohr magneton
gNV = 2.0028 # lande factor
h = 6.626e-34 # plancks constant
gammaNV = muB * gNV/h  # NV gyromagnetic ratio 
mug = muB * gNV
pi = np.pi
factor = (2*pi/h)**2 # normalization factor in fermis rule
W_psat = 1.9e7 # saturation excitation rate
c = 3e8
lambda_ = 532e-9 # wavelength of green lazer
tau_1 = 1e3 # spin conserved relaxation rate

k_init = 1e6 * np.array([[0, 0, 0, 0, 0, 0, 0],  # inital rates taken from Tetienne
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [63.2, 0, 0, 0, 0, 0, 10.8],
                             [0, 63.2, 0, 0, 0, 0, 60.7],
                             [0, 0, 63.2, 0, 0, 0, 60.7],
                             [0.8, 0.4, 0.4, 0, 0, 0, 0]])


# Pauli matrices
S_x = 1 / np.sqrt(2) * np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])
S_y = 1 / np.sqrt(2) * 1j * np.array([[0, -1, 0],
                                      [1, 0, -1],
                                      [0, 1, 0]])
S_z = np.array([[1, 0, 0],
                [0, 0, 0], 
                [0, 0, -1]])
SI = np.eye(3)

S_z_sq = np.dot(S_z, S_z)  # Sz^2


def Hamiltonian(B):  # Hamiltonian for excited states
    HZFS = h * (D_0) * S_z_sq + (mug * B[2] * S_z)  # Zero-field splitting
    HBEl = mug * (B[0] * S_x + B[1] * S_y)  # Zeeman coupling
    H_total = HZFS + HBEl
    # Eigenvalues and Eigenvectors and of full Hamiltonian
    E1, V1 = np.linalg.eigh(H_total)
    E2, V2 = np.linalg.eigh(HZFS)  # Eigenvalues of original Hamiltonian
    E1 = - min(E1) + E1  # return eigenvalues with respect to lowest

    return E1 / h, V1, V2  # convert eigenvalues to frequencies

def Hamiltonian_Exc(B):  # Hamiltonian for excited states
    HZFS = h * (D_E) * S_z_sq + (mug * B[2] * S_z)  # Zero-field splitting
    HBEl = mug * (B[0] * S_x + B[1] * S_y)  # Zeeman coupling
    H_total = HZFS + HBEl
    # Eigenvalues and Eigenvectors and of full Hamiltonian
    E1, V1 = np.linalg.eigh(H_total)
    E2, V2 = np.linalg.eigh(HZFS)  # Eigenvalues of original Hamiltonian
    E1 = - min(E1) + E1  # return eigenvalues with respect to lowest

    return E1 / h, V1, V2  # convert eigenvalues to frequencies



def Interaction_Hamiltonian(Bmw):
    HintEl = mug * (Bmw[0]*S_x + Bmw[1]*S_y + Bmw[2]*S_z)  # perturbation hamiltonian for microwave input

    return HintEl


def transition_strength(B, bmw):
    hint = Interaction_Hamiltonian(bmw) 
    e, v, h = Hamiltonian(B)

    t_01 = np.dot( (np.dot(v[:,1].transpose(), hint)) , v[:,0])
    ta_01 = np.abs(t_01) ** 2
    ts_01 = ta_01 * factor # transition strength from ground to (-)

    t_02 = np.dot( (np.dot(v[:,2].transpose(), hint)) , v[:,0])
    ta_02 = np.abs(t_02) ** 2
    ts_02 = ta_02 * factor   # transition strength from ground to (+)

    #print("TS1 = ",ts_01, " Hz, ", " TS2 = " , ts_02, "Hz")

    return ts_01, ts_02


def lorentzian(x, x0, d):
    L = (d/2)/((x-x0)**2 + (d/2)**2)  # lorentzian lineshape with input as center and linewidth

    return L

def alpha_matrix(B):
    e1, v1, v1_0 = Hamiltonian(B)
    e2, v2, v2_0 = Hamiltonian_Exc(B)
    v1_comb = np.hstack((v1, v2))
    v0_comb = np.hstack((v1_0, v2_0))
    v1_comb_conj_T = v1_comb.conj().T
    alpha = np.eye(7, dtype=complex)

    alpha[:6, :6] = np.diag(
        [np.dot(v1_comb_conj_T[i, :], v0_comb[:, i]) for i in range(6)])
    alpha[0, 1] = np.dot(v1_comb_conj_T[0], v0_comb[:, 1])
    alpha[0, 2] = np.dot(v1_comb_conj_T[0], v0_comb[:, 2])
    alpha[3, 4] = np.dot(v1_comb_conj_T[3], v0_comb[:, 4])
    alpha[3, 5] = np.dot(v1_comb_conj_T[3], v0_comb[:, 5])
    alpha[1, 0] = np.dot(v1_comb_conj_T[1], v0_comb[:, 0])
    alpha[2, 0] = np.dot(v1_comb_conj_T[2], v0_comb[:, 0])
    alpha[4, 3] = np.dot(v1_comb_conj_T[4], v0_comb[:, 3])
    alpha[5, 3] = np.dot(v1_comb_conj_T[5], v0_comb[:, 3])

    return alpha


def new_rates(B, beta, t1, t2, alpha_mat):  # calculation of rate matrix

    k_init[:3, 3:6] = beta * k_init[3:6, :3]
    k_new = np.zeros((7, 7))
    alpha = alpha_mat
    alpha_abs2 = np.abs(alpha)**2
    # calculation of new rates due to magnetic field
    k_new = np.einsum('ip,jq,pq->ij', alpha_abs2, alpha_abs2, k_init)
    k_new[0][1] = k_new[1][0] = t1  # additional rates due to perturbation
    k_new[0][2] = k_new[2][0] = t2

    return k_new


def population_matrix(B, beta, t1, t2, alpha): # creating population matrix
    k = new_rates(B, beta, t1, t2, alpha) # k is a 7x7 matrix #
    p = np.random.rand(8,7) # Over determined 8x7 equation

    p[:7, :] = k.T
    p[7, :] = 1
    row_sums = np.sum(k, axis=1)
    p[np.arange(7), np.arange(7)] -= row_sums

    return p


def population_equation(B, beta, t1, t2, alpha):
    rate_matrix = population_matrix(B, beta, t1, t2, alpha)
    sol = np.array([0,0,0,0,0,0,0,1])

    res = lsq_linear(rate_matrix, sol, bounds=(0, np.inf)) # solving for steady state population values with boundary from 0 to inf
    
    if res.success:
        x = res.x
        return x
    else:
        print("Optimization failed:", res.message)
        return None


def PL(B, beta, eta, t1, t2, alpha): # calculating photoluminescence
    k = new_rates(B, beta, t1, t2, alpha)
    n = population_equation(B, beta, t1, t2, alpha)
    R = np.sum(n[3:6][:, np.newaxis] * k[3:6, :3])
            
    return R * eta

def contrast_amp(B, beta, eta, bmw, sfreq, ffreq, nfreq, delta):
    MWfreq = np.linspace(sfreq, ffreq, nfreq)
    c = np.zeros(nfreq)
    e, v, h = Hamiltonian(B)
    print("E.V - ", e, " Hz")
    t1, t2 = transition_strength(B, bmw)
    alpha = alpha_matrix(B)

    t1 = t1 * lorentzian(MWfreq, e[1], delta) # creating lorentzian form for transitino strengths
    t2 = t2 * lorentzian(MWfreq, e[2], delta)
    c0 = PL(B, beta, eta, 0, 0, alpha)
    print("C0 = ", c0, "Counts/s")

    for i in np.arange(nfreq):
       c[i] = (c0 - PL(B, beta, eta, t1[i], t2[i], alpha) ) / c0 # looping over all transition strengths
    
    min1, min2 = sorted(1-c)[:2]
    print("Two dips -", min1, min2)
    
    return 1 - c # contrast calculation


def linewidth(P, sigma, w, B):

    I_sat = W_psat * c * h /sigma / lambda_ # saturation intensity of green lazer
    Psat  = pi * w**2 * I_sat / 2
    s = P/Psat # power factor
    beta = sigma * lambda_ / (63.2e6 * h * c) * (2 * P / pi / (w**2)) # pumping factor
    
    tau_c = 63.2e6 * s/(1+s) #change of k_0 with power
    tau_p = 5e6  * s/(1+s) #change of optical cycle time with power
    omega_r = mug / h * B #rabi frequency
    delta = 1/(2*pi) * np.sqrt((tau_c **2) + ((omega_r**2) * tau_c)/(2*tau_1 + tau_p)) #linewidth calculation
    
    print("Linewdith = ", delta/1e6, " Mhz", " \nbeta = ", beta)

    return delta, beta


def plot(B, eta, sfreq, ffreq, nfreq, P_mw, P_laz, sigma, w):
    B_mw = np.sqrt(2 *  377 * (10 **(P_mw/10))/1000)/c # converting power of microwave to B vec
    bx_mw= np.sqrt((B_mw**2)/2) # assume mw is present in X-Y plane only
    print("MW mag - ", bx_mw, "T")

    delta, beta = linewidth(P_laz, sigma, w, B_mw)

    C = contrast_amp(B, beta, eta, (bx_mw,bx_mw, 0), sfreq, ffreq, nfreq, delta)  # constrast
    MWfreq = np.linspace(sfreq, ffreq, nfreq)
    dC_dMWfreq = np.gradient(C, MWfreq)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8)) # plotting ODMR spectrum
    axs[0].plot(MWfreq, C, color='tab:green')
    axs[0].set_title('ODMR Spectrum')
    axs[0].set_xlabel('Microwave Frequency (Hz)')
    axs[0].set_ylabel('Contrast (a.u.)')
    axs[0].grid(True)

    axs[1].plot(MWfreq, dC_dMWfreq, color='tab:red')  # Plotting the differential ODMR spectrum
    axs[1].set_title('Differential ODMR Spectrum')
    axs[1].set_xlabel('Microwave Frequency (Hz)')
    axs[1].set_ylabel('d(Contrast)/d(Frequency)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


x = plot((0,0, 10e-4), 0.8, 2.83e9, 2.91e9, 2500, 15, 1, 9.1e-21, 10e-5)
# input format - ( MW coordinates in cartesian , efficiency of photon collector, start frequency of sweep,...
# end frequency of sweep, number of frequency points in sweep, Power of the MW, Power of the green lazer, area of NV centre, diameter of green lazer)









 