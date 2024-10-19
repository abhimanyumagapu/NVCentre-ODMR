import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import multiprocessing as mp
import time


# Constants
# NV fine constants

D_0 = 2.8700e9  # zero field splitting for ground state
D_E = 1.4200e9  # zero field splitting for excited state
muB = 9.2740e-24  # bohr magneton
gNV = 2.0028  # lande factor
h = 6.6260e-34  # plancks constant
mug = muB * gNV
pi = np.pi
factor = (2*pi/h)**2  # normalization factor in fermi's rule
W_psat = 1.9000e7  # saturation excitation rate
c = 2.9979e8
lambda_ = 532e-9  # wavelength of green lazer
tau_1 = 1e3  # spin conserved relaxation rate

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


def Hamiltonian(B):  # Hamiltonian for static magnetic field
    # Zero-field splitting, Original Hamilonian
    HZFS = h * (D_0) * S_z_sq + (mug * B[2] * S_z)
    HBEl = mug * (B[0] * S_x + B[1] * S_y)  # Zeeman coupling, Perturbation
    H_total = HZFS + HBEl
    # Eigenvalues and Eigenvectors and of full Hamiltonian
    E1, V1 = np.linalg.eigh(H_total)
    E2, V2 = np.linalg.eigh(HZFS)  # Eigenvalues of original Hamiltonian
    E1 = - min(E1) + E1  # return eigenvalues with respect to lowest

    return E1/h, V1, V2  # convert eigenvalues to frequencies


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
    # perturbation hamiltonian for microwave input
    HintEl = mug * (Bmw[0]*S_x + Bmw[1]*S_y + Bmw[2]*S_z)

    return HintEl


def transition_strength(B, bmw):
    hint = Interaction_Hamiltonian(bmw)
    e, v, v_0 = Hamiltonian(B)
    t_01 = np.dot((np.dot(v[:, 1].conj().transpose(), hint)), v[:, 0])
    ta_01 = np.abs(t_01) ** 2
    ts_01 = ta_01 * factor  # transition strength from ground to (-)

    t_02 = np.dot((np.dot(v[:, 2].conj().transpose(), hint)), v[:, 0])
    ta_02 = np.abs(t_02) ** 2
    ts_02 = ta_02 * factor   # transition strength from ground to (+)

    return ts_01, ts_02


def lorentzian(x, x0, d):
    # lorentzian lineshape with input as center and linewidth
    L = (d/2)/((x-x0)**2 + (d/2)**2)

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


def MW_B(P_mw):  # Returns magnetic vector from power in dbm

    return np.sqrt(2 * 377 * (10 ** (P_mw/10))/1000)/c


def population_matrix(B, beta, t1, t2, alpha):  # creating population matrix
    k = new_rates(B, beta, t1, t2, alpha)  # k is a 7x7 matrix #
    p = np.zeros((8, 7))  # Over determined 8x7 equation

    p[:7, :] = k.T
    p[7, :] = 1
    row_sums = np.sum(k, axis=1)
    p[np.arange(7), np.arange(7)] -= row_sums

    return p


def population_equation(B, beta, t1, t2, alpha):
    rate_matrix = population_matrix(B, beta, t1, t2, alpha)
    sol = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    # solving for steady state population values with boundary from 0 to inf
    res = lsq_linear(rate_matrix, sol, bounds=(0, np.inf))
    if res.success:
        x = res.x
        return x
    else:
        print("Optimization failed:", res.message)
        return None


def PL(B, beta, eta, t1, t2, alpha):  # calculating photoluminescence
    k = new_rates(B, beta, t1, t2, alpha)
    n = population_equation(B, beta, t1, t2, alpha)
    R = np.sum(n[3:6][:, np.newaxis] * k[3:6, :3])

    return R * eta


def linewidth(P, sigma, w, P_mw):
    I_sat = W_psat * c * h / sigma / lambda_  # saturation intensity of green lazer
    Psat = pi * w**2 * I_sat / 2
    s = P/Psat  # power factor
    beta = sigma * lambda_ / (63.2e6 * h * c) * (2 * P / pi / (w**2))  # pumping factor

    tau_c = 63.2e6 * s/(1+s)  # change of k_0 with power
    tau_p = 5e6 * s/(1+s)  # change of optical cycle time with power
    B = MW_B(P_mw)
    omega_r = mug / h * B  # rabi frequency
    delta = 1/(2*pi) * np.sqrt((tau_c ** 2) + ((omega_r**2) *
                                               tau_c)/(2*tau_1 + tau_p))  # linewidth calculation

    #print("Linewdith = ", delta/1e6, " Mhz", " \nbeta = ", beta, "\n")

    return delta, beta


def get_vector_cartesian(A, theta, phi):
    vec = np.array([A * np.sin(theta) * np.cos(phi),
                    A * np.sin(theta) * np.sin(phi),
                    A * np.cos(theta)])
    return vec


# Transformation between lab frame and NV frames, cartesian coordinates
def get_rotation_matrix(idx_nv):
    if idx_nv == 1:
        RNV = np.array([[0, 1, 0],
                        [-np.sqrt(1/3), 0, np.sqrt(2/3)],
                        [np.sqrt(2/3), 0, np.sqrt(1/3)]])
    elif idx_nv == 2:
        RNV = np.array([[1, 0, 0],
                        [0, np.sqrt(1/3), -np.sqrt(2/3)],
                        [0, -np.sqrt(2/3), -np.sqrt(1/3)]])
    elif idx_nv == 3:
        RNV = np.array([[1, 0, 0],
                        [0, -np.sqrt(1/3), -np.sqrt(2/3)],
                        [0, np.sqrt(2/3), -np.sqrt(1/3)]])
    elif idx_nv == 4:
        RNV = np.array([[0, 1, 0],
                        [np.sqrt(1/3), 0, np.sqrt(2/3)],
                        [-np.sqrt(2/3), 0, np.sqrt(1/3)]])
    else:
        raise ValueError('Invalid index of NV orientation')

    return RNV


def transform_vector_lab_to_NV_frame(vec_in_lab, nv_idx):
    RNV = get_rotation_matrix(nv_idx)
    vec_in_nv = np.dot(RNV, vec_in_lab)
    return vec_in_nv


def transform_all_frames(Bvec):
    Bvec_list = [transform_vector_lab_to_NV_frame(
        Bvec, idx) for idx in range(1, 5)]

    return Bvec_list


def transform_all_frames_spherical(B0, theta, phi):
    Bvec = get_vector_cartesian(B0, theta, phi)
    Bvec_list = [transform_vector_lab_to_NV_frame(
        Bvec, idx) for idx in range(1, 5)]

    return Bvec_list


def contrast_amp_ensemble_withT(Bvec, beta, eta, P_mw, thetaMW, phiMW, sfreq, ffreq, nfreq, delta):
    Bvector_list = transform_all_frames(Bvec)

    B_mw = MW_B(P_mw)
    MWvector_list = transform_all_frames_spherical(B_mw, thetaMW, phiMW)
    MWfreq = np.linspace(sfreq, ffreq, nfreq)

    c_ampN = []
    c_ampV = []
    c_amp = []
    Pl = []
    PlV = []
    PlN = []
    alpha = np.array([alpha_matrix(B) for B in Bvector_list])

    for i in range(4):  # for NV
        #print(i, "\n")
        c = np.zeros(nfreq)
        e, v, v_0 = Hamiltonian(Bvector_list[i])

        #print("E.V (NV) - ", e, " Hz")
        t1, t2 = transition_strength(Bvector_list[i], MWvector_list[i])
        #print("TS1 (NV) =", t1, " Hz, ", " TS2(NV) = ", t2, "Hz")

        # creating lorentzian form for transition strengths
        t1 = t1 * lorentzian(MWfreq, e[1], delta)
        t2 = t2 * lorentzian(MWfreq, e[2], delta)
        c0 = PL(Bvector_list[i], beta, eta, 0, 0, alpha[i])
        Pl.append(c0)
        #print("C0 = ", c0, "Counts/s")
        #print("\n")
        for j in np.arange(nfreq):
            # looping over all transition strengths
            c[j] = PL(Bvector_list[i], beta, eta, t1[j], t2[j], alpha[i])
        c_amp.append(c)

    #print("-------- \n")

    for i in range(4):  # for VN
        #print(i, "\n")
        Bvector_list[i][2] = -Bvector_list[i][2]
        c = np.zeros(nfreq)
        e, v, v_0 = Hamiltonian(Bvector_list[i])
        #print("E.V (VN) - ", e, " Hz")
        t1, t2 = transition_strength(Bvector_list[i], MWvector_list[i])
        #print("TS1 (VN) =", t1, " Hz, ", " TS2(VN) = ", t2, "Hz")

        # creating lorentzian form for transition strengths
        t1 = t1 * lorentzian(MWfreq, e[1], delta)
        t2 = t2 * lorentzian(MWfreq, e[2], delta)
        c0 = PL(Bvector_list[i], beta, eta, 0, 0, alpha[i])
        Pl.append(c0)
        #print("C0 = ", c0, "Counts/s")
        #print("\n")
        for j in np.arange(nfreq):
            # looping over all transition strengths
            c[j] = PL(Bvector_list[i], beta, eta, t1[j], t2[j], alpha[i])
        c_amp.append(c)

    return Pl, c_amp # contrast calculation


def ODMR_ensemble(Bvec, beta, eta, P_mw, thetaMW, phiMW, sfreq, ffreq, nfreq, delta):

    pl, c= contrast_amp_ensemble_withT(
        Bvec, beta, eta, P_mw, thetaMW, phiMW, sfreq, ffreq, nfreq, delta)
    
    C0_total = sum(pl)
    C_amp_total = sum(c)
    c = (C0_total - C_amp_total) / C0_total

    #print("\nPeaks", sorted(1-c)[:8])

    return 1-c


def plot(Bvec, eta, sfreq, ffreq, nfreq, P_mw, thetaMW, phiMW, P_laz, sigma, w, filename):
    delta, beta = linewidth(P_laz, sigma, w, P_mw)

    C = ODMR_ensemble(Bvec, beta, eta, P_mw, thetaMW, phiMW,
                      sfreq, ffreq, nfreq, delta)  # constrast
    MWfreq = np.linspace(sfreq, ffreq, nfreq)
    dC_dMWfreq = np.gradient(C, MWfreq)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(MWfreq, C, color='tab:green', label = 'C')
    axs[0].set_title(f'Plot for Bvec: {Bvec}')
    axs[0].set_xlabel('Microwave Frequency (Hz)')
    axs[0].set_ylabel('Contrast (a.u.)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(MWfreq, dC_dMWfreq, color='tab:red')
    axs[1].set_title('Differential ODMR Spectrum')
    axs[1].set_xlabel('Microwave Frequency (Hz)')
    axs[1].set_ylabel('d(Contrast)/d(Frequency)')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()
  


def parallel_plot(args):
       Bvec, eta, sfreq, ffreq, nfreq, P_mw, thetaMW, phiMW, P_laz, sigma, w, filename = args
       return plot(Bvec, eta, sfreq, ffreq, nfreq, P_mw, thetaMW, phiMW, P_laz, sigma, w, filename)

def paral_plot(B):

    eta = 1
    sfreq = 2.83e9
    ffreq = 2.91e9
    nfreq = 2000
    P_mw = 50
    thetaMW = np.pi/2
    phiMW = np.pi/4
    P_laz = 1
    sigma = 9.1e-21
    w = 10e-5

    args_list = [
        (Bvec, eta, sfreq, ffreq, nfreq, P_mw, thetaMW, phiMW, P_laz, sigma, w, f'plot_Bvec{i}{j}') 
        for i in range(B.shape[0]) for j in range(B.shape[1]) for Bvec in [B[i][j]]
    ]

    if __name__ == '__main__' : 
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(parallel_plot, args_list)
    return 0


low = 5e-4
high = 20e-4

# Generate a 100x100 array of lists with 3 random elements each
B = np.random.uniform(low, high, (10,10, 3))
#y = paral_plot(B)
x = plot([7e-4,6e-4,4e-4], 1, 2.83e9, 2.91e9, 2500, 50, pi / 2, pi / 4, 1, 9.1e-21, 10e-5, 'test')





