import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, void, float64, int64

# ======================================================================
# 1. NUMBA COMPLIED FUNCTION (C-like performance)
# ======================================================================
# PAPER/SM to CODE MAP: This functions solves the ODE system
# outlined in sections SM.B.1, SM.B.2, SM.B.3 e SM.B.4 [SM].
# We are interesterd only in risk free cumulants, \alpha^(1) [SM.B.1] does not affect them,
# so we do not calculate it

# Numba array pointer definition
f64 = float64
arr1d = float64[::1]       # 1D array C-contigous
arr2d = float64[:, ::1]    # 2D matrix C-contigous
arr3d = float64[:, :, ::1] # 3D tensor C-contigous

# Exact firm of all input variables (parameters). First parameter is the "return" type (void)
@njit(void(
    f64, arr1d, arr2d, arr1d, arr1d, 
    arr1d, arr2d, arr3d, 
    f64, arr1d, arr2d, 
    arr1d, arr2d, arr3d, arr2d, arr2d, arr2d, arr3d, 
    f64, arr1d, arr2d, arr1d, arr1d, arr1d, arr2d, 
    arr1d, arr2d, arr3d, arr2d, arr2d, arr3d, arr3d, 
    arr2d, arr2d, arr2d, 
    arr2d, arr3d, arr3d, 
    arr2d, arr3d, 
    f64, arr1d, arr2d, arr1d, arr1d, arr2d, arr2d, 
    arr1d, arr1d, arr1d, 
    arr1d, arr2d, arr2d, 
    arr1d, arr2d,
    arr1d        # <- dY_out buffer linked by reference
), fastmath=True, cache=True)
def fast_ode_system(tau, Y, A, B1, A_alpha, # end general/ 1° cumulant
                    B2_c, B2_s, B2_q, #
                    alpha_c_2, alpha_s_2, alpha_q_2, # end 2° cumulant 
                    B3_c, B3_s, B3_q_c, 
                    B3_q_s_1, B3_q_s_3, B3_h_c, B3_h_s, #
                    alpha_c_3, alpha_s_3, alpha_q_c, alpha_q_s_1, alpha_q_s_3, alpha_h_c, alpha_h_s, # end 3° cumulant
                    B4_c, B4_s, B4_q_c, B4_q_s_1_4, B4_q_s_3_4, B4_q_q_1, B4_q_q_3, 
                    B4_q_s_h_1, B4_q_s_h_3, B4_q_s_h_12, 
                    B4_h_c, B4_h_s, B4_h_q, 
                    B4_l_c, B4_l_s, #
                    alpha_c_4, alpha_s_4, alpha_q_c_4, alpha_q_s_4_1, alpha_q_s_4_3, alpha_q_q_4_1, alpha_q_q_4_3,
                    alpha_q_s_h_4_1, alpha_q_s_h_4_3, alpha_q_s_h_4_1_2,
                    alpha_h_c_4, alpha_h_s_4, alpha_h_q_4,
                    alpha_l_c_4, alpha_l_s_4, # end 4° cumulant
                    dY_out):
    
    # ==========================================================
    # 1° CUMULANT (Vector beta1 = Y[0:3])
    # ==========================================================
    for i in range(3):
        dot_A = A[i, 0]*Y[0] + A[i, 1]*Y[1] + A[i, 2]*Y[2] # A @ beta1
        dY_out[i] = B1[i] + dot_A
    
    # ==========================================================
    # 2° CUMULANT (Vector beta2 = Y[3:6] and Intercept alpha2)
    # ==========================================================
    for j in range(3):
        dot_s = B2_s[j, 0]*Y[0] + B2_s[j, 1]*Y[1] + B2_s[j, 2]*Y[2] # B_{j, s}^(2) @ beta1
        dot_q = 0.0
        for r in range(3):
            for c in range(3): dot_q += Y[r] * B2_q[j, r, c] * Y[c] # beta1 @ B_{j, q}^(2) @ beta1
        dot_A = A[j, 0]*Y[3] + A[j, 1]*Y[4] + A[j, 2]*Y[5]          # A @ beta2
        dY_out[3+j] = B2_c[j] + dot_s + dot_q + dot_A               # complete equation

    dot_s_a2 = alpha_s_2[0]*Y[0] + alpha_s_2[1]*Y[1] + alpha_s_2[2]*Y[2] # alpha_{s}^(2) @ beta1
    dot_q_a2 = 0.0
    for r in range(3):
        for c in range(3): dot_q_a2 += Y[r] * alpha_q_2[r, c] * Y[c]     # beta1 @ alpha_q^(2) @ beta1
    dot_A_a2 = A_alpha[0]*Y[3] + A_alpha[1]*Y[4] + A_alpha[2]*Y[5]       # A_aplha * beta2
    dY_out[12] = alpha_c_2 + dot_s_a2 + dot_q_a2 + dot_A_a2
    
    # ==========================================================
    # 3° CUMULANT (Vector beta3 = Y[6:9] and Intercept alpha3)
    # ==========================================================
    for j in range(3):
        dot_s = B3_s[j, 0]*Y[0] + B3_s[j, 1]*Y[1] + B3_s[j, 2]*Y[2]
        dot_q = 0.0
        for r in range(3):
            for c in range(3): 
                dot_q += Y[r] * (B3_q_c[j, r, c]) * Y[c] # only the constant terms of beta1 @ (B3_q_c + termq_s) @ beta1
            
        dot_q_s_1 = B3_q_s_1[j, 0]*Y[0] + B3_q_s_1[j, 1]*Y[1] + B3_q_s_1[j, 2]*Y[2]
        dot_q_s_3 = B3_q_s_3[j, 0]*Y[0] + B3_q_s_3[j, 1]*Y[1] + B3_q_s_3[j, 2]*Y[2]
        term_q_s = Y[0]*dot_q_s_1*Y[0] + Y[1]*dot_q_s_3*Y[1]
        
        dot_h_c = B3_h_c[j, 0]*Y[3] + B3_h_c[j, 1]*Y[4] + B3_h_c[j, 2]*Y[5] # constant term of _ @ beta2
        dot_h_s = 0.0
        for r in range(3):
            row_val = B3_h_s[j, r, 0]*Y[0] + B3_h_s[j, r, 1]*Y[1] + B3_h_s[j, r, 2]*Y[2]
            dot_h_s += row_val * Y[3+r] # time dependent term of _ @ beta2
            
        dot_A = A[j, 0]*Y[6] + A[j, 1]*Y[7] + A[j, 2]*Y[8]
        dY_out[6+j] = B3_c[j] + dot_s + dot_q + term_q_s + dot_h_c + dot_h_s + dot_A

    #alpha3
    dot_s_a3 = alpha_s_3[0]*Y[0] + alpha_s_3[1]*Y[1] + alpha_s_3[2]*Y[2]

    dot_q_a3 = 0.0
    for r in range(3):
        for c in range(3): dot_q_a3 += Y[r] * alpha_q_c[r, c] * Y[c]

    dot_q_s_1_a3 = alpha_q_s_1[0]*Y[0] + alpha_q_s_1[1]*Y[1] + alpha_q_s_1[2]*Y[2]
    dot_q_s_3_a3 = alpha_q_s_3[0]*Y[0] + alpha_q_s_3[1]*Y[1] + alpha_q_s_3[2]*Y[2]
    term_q_s_a3 = Y[0]*dot_q_s_1_a3*Y[0] + Y[1]*dot_q_s_3_a3*Y[1]

    dot_h_c_a3 = alpha_h_c[0]*Y[3] + alpha_h_c[1]*Y[4] + alpha_h_c[2]*Y[5]

    dot_h_s_a3 = 0.0
    for r in range(3):
        row_val = alpha_h_s[r, 0]*Y[0] + alpha_h_s[r, 1]*Y[1] + alpha_h_s[r, 2]*Y[2]
        dot_h_s_a3 += row_val * Y[3+r]

    dot_A_a3 = A_alpha[0]*Y[6] + A_alpha[1]*Y[7] + A_alpha[2]*Y[8]
    dY_out[13] = alpha_c_3 + dot_s_a3 + dot_q_a3 + term_q_s_a3 + dot_h_c_a3 + dot_h_s_a3 + dot_A_a3
    
    # ==========================================================
    # 4° CUMULANT (Vector beta4 = Y[9:12] e Intercept alpha4)
    # ==========================================================
    for j in range(3):
        dot_s = B4_s[j, 0]*Y[0] + B4_s[j, 1]*Y[1] + B4_s[j, 2]*Y[2]
        
        dot_q_c = 0.0
        for r in range(3):
            for c in range(3): dot_q_c += Y[r] * B4_q_c[j, r, c] * Y[c]
        #
        term_41 = B4_q_s_1_4[j, 0]*Y[0] + B4_q_s_1_4[j, 1]*Y[1] + B4_q_s_1_4[j, 2]*Y[2]
        term_43 = B4_q_s_3_4[j, 0]*Y[0] + B4_q_s_3_4[j, 1]*Y[1] + B4_q_s_3_4[j, 2]*Y[2]
        term_412 = B4_q_s_h_12[j, 0]*Y[3] + B4_q_s_h_12[j, 1]*Y[4] + B4_q_s_h_12[j, 2]*Y[5]
        
        for r in range(3):
            for c in range(3):
                term_41 += Y[r] * B4_q_q_1[j, r, c] * Y[c]
                term_43 += Y[r] * B4_q_q_3[j, r, c] * Y[c]
                
        term_41 += B4_q_s_h_1[j, 0]*Y[3] + B4_q_s_h_1[j, 1]*Y[4] + B4_q_s_h_1[j, 2]*Y[5]
        term_43 += B4_q_s_h_3[j, 0]*Y[3] + B4_q_s_h_3[j, 1]*Y[4] + B4_q_s_h_3[j, 2]*Y[5]
        
        term_q_s_tau = Y[0]*term_41*Y[0] + Y[1]*term_43*Y[1] + 2.0*Y[0]*term_412*Y[1]
        #
        dot_h_c = B4_h_c[j, 0]*Y[3] + B4_h_c[j, 1]*Y[4] + B4_h_c[j, 2]*Y[5]
        dot_h_s = 0.0
        for r in range(3):
            row_val = B4_h_s[j, r, 0]*Y[0] + B4_h_s[j, r, 1]*Y[1] + B4_h_s[j, r, 2]*Y[2]
            dot_h_s += row_val * Y[3+r]
            
        dot_h_q = 0.0
        for r in range(3):
            for c in range(3): dot_h_q += Y[3+r] * B4_h_q[j, r, c] * Y[3+c]
            
        dot_l_c = B4_l_c[j, 0]*Y[6] + B4_l_c[j, 1]*Y[7] + B4_l_c[j, 2]*Y[8]
        dot_l_s = 0.0
        for r in range(3):
            row_val = B4_l_s[j, r, 0]*Y[0] + B4_l_s[j, r, 1]*Y[1] + B4_l_s[j, r, 2]*Y[2]
            dot_l_s += row_val * Y[6+r]
            
        dot_A = A[j, 0]*Y[9] + A[j, 1]*Y[10] + A[j, 2]*Y[11]
        dY_out[9+j] = B4_c[j] + dot_s + dot_q_c + term_q_s_tau + dot_h_c + dot_h_s + dot_h_q + dot_l_c + dot_l_s + dot_A

    #alpha4
    dot_s_a4 = alpha_s_4[0]*Y[0] + alpha_s_4[1]*Y[1] + alpha_s_4[2]*Y[2]
    
    dot_q_c_a4 = 0.0
    for r in range(3):
        for c in range(3): dot_q_c_a4 += Y[r] * alpha_q_c_4[r, c] * Y[c]
    #
    aterm_41 = alpha_q_s_4_1[0]*Y[0] + alpha_q_s_4_1[1]*Y[1] + alpha_q_s_4_1[2]*Y[2]
    aterm_43 = alpha_q_s_4_3[0]*Y[0] + alpha_q_s_4_3[1]*Y[1] + alpha_q_s_4_3[2]*Y[2]
    aterm_412 = alpha_q_s_h_4_1_2[0]*Y[3] + alpha_q_s_h_4_1_2[1]*Y[4] + alpha_q_s_h_4_1_2[2]*Y[5]
    
    for r in range(3):
        for c in range(3):
            aterm_41 += Y[r] * alpha_q_q_4_1[r, c] * Y[c]
            aterm_43 += Y[r] * alpha_q_q_4_3[r, c] * Y[c]
            
    aterm_41 += alpha_q_s_h_4_1[0]*Y[3] + alpha_q_s_h_4_1[1]*Y[4] + alpha_q_s_h_4_1[2]*Y[5]
    aterm_43 += alpha_q_s_h_4_3[0]*Y[3] + alpha_q_s_h_4_3[1]*Y[4] + alpha_q_s_h_4_3[2]*Y[5]
    
    aterm_q_s_tau = Y[0]*aterm_41*Y[0] + Y[1]*aterm_43*Y[1] + 2.0*Y[0]*aterm_412*Y[1]
    #
    dot_h_c_a4 = alpha_h_c_4[0]*Y[3] + alpha_h_c_4[1]*Y[4] + alpha_h_c_4[2]*Y[5]
    dot_h_s_a4 = 0.0
    for r in range(3):
        row_val = alpha_h_s_4[r, 0]*Y[0] + alpha_h_s_4[r, 1]*Y[1] + alpha_h_s_4[r, 2]*Y[2]
        dot_h_s_a4 += row_val * Y[3+r]
        
    dot_h_q_a4 = 0.0
    for r in range(3):
        for c in range(3): dot_h_q_a4 += Y[3+r] * alpha_h_q_4[r, c] * Y[3+c]
        
    dot_l_c_a4 = alpha_l_c_4[0]*Y[6] + alpha_l_c_4[1]*Y[7] + alpha_l_c_4[2]*Y[8]
    dot_l_s_a4 = 0.0
    for r in range(3):
        row_val = alpha_l_s_4[r, 0]*Y[0] + alpha_l_s_4[r, 1]*Y[1] + alpha_l_s_4[r, 2]*Y[2]
        dot_l_s_a4 += row_val * Y[6+r]
        
    dot_A_a4 = A_alpha[0]*Y[9] + A_alpha[1]*Y[10] + A_alpha[2]*Y[11]
    dY_out[14] = alpha_c_4 + dot_s_a4 + dot_q_c_a4 + aterm_q_s_tau + dot_h_c_a4 + dot_h_s_a4 + dot_h_q_a4 + dot_l_c_a4 + dot_l_s_a4 + dot_A_a4

    # No return needed: dY_out passed by reference.

# ======================================================================
# 1.5 RUNGE-KUTTA 4 INTEGRATOR WITH FIXED STEP
# ======================================================================
@njit(arr1d(
    f64, arr1d, int64, # tau, Y0, steps
    arr2d, arr1d, arr1d, 
    arr1d, arr2d, arr3d, 
    f64, arr1d, arr2d, 
    arr1d, arr2d, arr3d, arr2d, arr2d, arr2d, arr3d,
    f64, arr1d, arr2d, arr1d, arr1d, arr1d, arr2d, 
    arr1d, arr2d, arr3d, arr2d, arr2d, arr3d, arr3d, 
    arr2d, arr2d, arr2d, 
    arr2d, arr3d, arr3d, 
    arr2d, arr3d, 
    f64, arr1d, arr2d, arr1d, arr1d, arr2d, arr2d, 
    arr1d, arr1d, arr1d, 
    arr1d, arr2d, arr2d, 
    arr1d, arr2d
), fastmath=True, cache=True)
def rk4_integrator(tau, Y0, steps,
                   A, B1, A_alpha, 
                   B2_c, B2_s, B2_q, 
                   alpha_c_2, alpha_s_2, alpha_q_2, 
                   B3_c, B3_s, B3_q_c, B3_q_s_1, B3_q_s_3, B3_h_c, B3_h_s, 
                   alpha_c_3, alpha_s_3, alpha_q_c, alpha_q_s_1, alpha_q_s_3, alpha_h_c, alpha_h_s,
                   B4_c, B4_s, B4_q_c, B4_q_s_1_4, B4_q_s_3_4, B4_q_q_1, B4_q_q_3, 
                   B4_q_s_h_1, B4_q_s_h_3, B4_q_s_h_12, 
                   B4_h_c, B4_h_s, B4_h_q, 
                   B4_l_c, B4_l_s, 
                   alpha_c_4, alpha_s_4, alpha_q_c_4, alpha_q_s_4_1, alpha_q_s_4_3, alpha_q_q_4_1, alpha_q_q_4_3,
                   alpha_q_s_h_4_1, alpha_q_s_h_4_3, alpha_q_s_h_4_1_2,
                   alpha_h_c_4, alpha_h_s_4, alpha_h_q_4,
                   alpha_l_c_4, alpha_l_s_4):
    
    # Inizialization
    Y_curr = np.copy(Y0)
    h = tau / float(steps)
    h_half = h * 0.5
    h_sixth = h / 6.0
    
    # Buffer pre-allocation (ZERO heap-allocation inside the loop)
    k1 = np.zeros(15, dtype=np.float64)
    k2 = np.zeros(15, dtype=np.float64)
    k3 = np.zeros(15, dtype=np.float64)
    k4 = np.zeros(15, dtype=np.float64)
    Y_tmp = np.zeros(15, dtype=np.float64)
    
    for i in range(steps):
        t_curr = i * h
        
        # --- k1 ---
        fast_ode_system(t_curr, Y_curr, A, B1, A_alpha, 
                        B2_c, B2_s, B2_q, alpha_c_2, alpha_s_2, alpha_q_2, 
                        B3_c, B3_s, B3_q_c, B3_q_s_1, B3_q_s_3, B3_h_c, B3_h_s, 
                        alpha_c_3, alpha_s_3, alpha_q_c, alpha_q_s_1, alpha_q_s_3, alpha_h_c, alpha_h_s,
                        B4_c, B4_s, B4_q_c, B4_q_s_1_4, B4_q_s_3_4, B4_q_q_1, B4_q_q_3, 
                        B4_q_s_h_1, B4_q_s_h_3, B4_q_s_h_12, B4_h_c, B4_h_s, B4_h_q, B4_l_c, B4_l_s, 
                        alpha_c_4, alpha_s_4, alpha_q_c_4, alpha_q_s_4_1, alpha_q_s_4_3, alpha_q_q_4_1, alpha_q_q_4_3,
                        alpha_q_s_h_4_1, alpha_q_s_h_4_3, alpha_q_s_h_4_1_2, alpha_h_c_4, alpha_h_s_4, alpha_h_q_4, alpha_l_c_4, alpha_l_s_4, 
                        k1)
        
        # --- k2 ---
        for d in range(15): Y_tmp[d] = Y_curr[d] + h_half * k1[d]
        fast_ode_system(t_curr + h_half, Y_tmp, A, B1, A_alpha, 
                        B2_c, B2_s, B2_q, alpha_c_2, alpha_s_2, alpha_q_2, 
                        B3_c, B3_s, B3_q_c, B3_q_s_1, B3_q_s_3, B3_h_c, B3_h_s, 
                        alpha_c_3, alpha_s_3, alpha_q_c, alpha_q_s_1, alpha_q_s_3, alpha_h_c, alpha_h_s,
                        B4_c, B4_s, B4_q_c, B4_q_s_1_4, B4_q_s_3_4, B4_q_q_1, B4_q_q_3, 
                        B4_q_s_h_1, B4_q_s_h_3, B4_q_s_h_12, B4_h_c, B4_h_s, B4_h_q, B4_l_c, B4_l_s, 
                        alpha_c_4, alpha_s_4, alpha_q_c_4, alpha_q_s_4_1, alpha_q_s_4_3, alpha_q_q_4_1, alpha_q_q_4_3,
                        alpha_q_s_h_4_1, alpha_q_s_h_4_3, alpha_q_s_h_4_1_2, alpha_h_c_4, alpha_h_s_4, alpha_h_q_4, alpha_l_c_4, alpha_l_s_4, 
                        k2)
        
        # --- k3 ---
        for d in range(15): Y_tmp[d] = Y_curr[d] + h_half * k2[d]
        fast_ode_system(t_curr + h_half, Y_tmp, A, B1, A_alpha, 
                        B2_c, B2_s, B2_q, alpha_c_2, alpha_s_2, alpha_q_2, 
                        B3_c, B3_s, B3_q_c, B3_q_s_1, B3_q_s_3, B3_h_c, B3_h_s, 
                        alpha_c_3, alpha_s_3, alpha_q_c, alpha_q_s_1, alpha_q_s_3, alpha_h_c, alpha_h_s,
                        B4_c, B4_s, B4_q_c, B4_q_s_1_4, B4_q_s_3_4, B4_q_q_1, B4_q_q_3, 
                        B4_q_s_h_1, B4_q_s_h_3, B4_q_s_h_12, B4_h_c, B4_h_s, B4_h_q, B4_l_c, B4_l_s, 
                        alpha_c_4, alpha_s_4, alpha_q_c_4, alpha_q_s_4_1, alpha_q_s_4_3, alpha_q_q_4_1, alpha_q_q_4_3,
                        alpha_q_s_h_4_1, alpha_q_s_h_4_3, alpha_q_s_h_4_1_2, alpha_h_c_4, alpha_h_s_4, alpha_h_q_4, alpha_l_c_4, alpha_l_s_4, 
                        k3)
        
        # --- k4 ---
        for d in range(15): Y_tmp[d] = Y_curr[d] + h * k3[d]
        fast_ode_system(t_curr + h, Y_tmp, A, B1, A_alpha, 
                        B2_c, B2_s, B2_q, alpha_c_2, alpha_s_2, alpha_q_2, 
                        B3_c, B3_s, B3_q_c, B3_q_s_1, B3_q_s_3, B3_h_c, B3_h_s, 
                        alpha_c_3, alpha_s_3, alpha_q_c, alpha_q_s_1, alpha_q_s_3, alpha_h_c, alpha_h_s,
                        B4_c, B4_s, B4_q_c, B4_q_s_1_4, B4_q_s_3_4, B4_q_q_1, B4_q_q_3, 
                        B4_q_s_h_1, B4_q_s_h_3, B4_q_s_h_12, B4_h_c, B4_h_s, B4_h_q, B4_l_c, B4_l_s, 
                        alpha_c_4, alpha_s_4, alpha_q_c_4, alpha_q_s_4_1, alpha_q_s_4_3, alpha_q_q_4_1, alpha_q_q_4_3,
                        alpha_q_s_h_4_1, alpha_q_s_h_4_3, alpha_q_s_h_4_1_2, alpha_h_c_4, alpha_h_s_4, alpha_h_q_4, alpha_l_c_4, alpha_l_s_4, 
                        k4)
        
        # Updating the state vector
        for d in range(15):
            Y_curr[d] += h_sixth * (k1[d] + 2.0*k2[d] + 2.0*k3[d] + k4[d])
            
    return Y_curr


# ======================================================================
# 2. MAIN CLASS
# ======================================================================
class AFTCumulantCalculator:
    def __init__(self, theta):
        self.p = theta
        
        # MAP PAPER Eq. 1-4: AFT model parameters assegnation[Paper].
        # Variances and stochastic equations for factors V_1, V_2, V_3
        self.k1, self.vbar1, self.sig1, self.rho1 = theta['k1'], theta['vbar1'], theta['sig1'], theta['rho1']
        self.k2, self.vbar2, self.sig2, self.rho2 = theta['k2'], theta['vbar2'], theta['sig2'], theta['rho2']
        self.k3, self.mu3, self.rho3 = theta['k3'], theta['mu3'], theta['rho3']
        
        # Jump distributional parameters
        self.mu1, self.lam_m, self.lam_p = theta['mu1'], theta['lam_m'], theta['lam_p']
        
        # Loadings factors for jump intensity 
        self.c0_m, self.c1_m, self.c2_m, self.c3_m = theta['c0_m'], theta['c1_m'], theta['c2_m'], theta['c3_m']
        self.c0_p, self.c1_p, self.c2_p, self.c3_p = theta['c0_p'], theta['c1_p'], theta['c2_p'], theta['c3_p']
        
        self.eta = theta['eta']
        
        # Init seqeunce
        self._precompute_jump_derivatives()
        self.A = self._get_matrix_A()
        self.B1 = self._get_vector_B1()
        
        # MAP SM.B.1 (and text): A_alpha = column vector alpha
        # A_alpha = [k1*vbar1 + c0_m * dTc_dq1, c0_m*(dTc_dq3 + dTi_dq3), k2*vbar2]^T[SM].
        self.A_alpha = np.array([
            self.k1 * self.vbar1 + self.c0_m * self.dTc_dq1, 
            self.c0_m * (self.dTc_dq3 + self.dTi_dq3), 
            self.k2 * self.vbar2
        ])
        
        # Pre-calculation to avoid memory waste
        self._precompute_ode_matrices()

    def _precompute_jump_derivatives(self):
        # MAP SM.B: functions \Theta_c^-, \Theta_i^-, \Theta^+ are computed 
        # as derivatives computed in u=0 (q0, q1, q3 = 0). Here we compute
        # analatical derivatives needed of the Exp component[SM].
        self.E_z_m = -1.0 / self.lam_m
        self.E_z2_m = 2.0 / (self.lam_m ** 2)
        self.E_z3_m = -6.0 / (self.lam_m ** 3)
        self.E_z4_m = 24.0 / (self.lam_m ** 4)
        self.E_z5_m = -120.0 / (self.lam_m ** 5)
        self.E_z6_m = 720.0 / (self.lam_m ** 6)
        self.E_z7_m = -5040.0 / (self.lam_m ** 7)
        self.E_z8_m = 40320.0 / (self.lam_m ** 8)

        self.E_z_p = 1.0 / self.lam_p
        self.E_z2_p = 2.0 / (self.lam_p ** 2)
        self.E_z3_p = 6.0 / (self.lam_p ** 3)
        self.E_z4_p = 24.0 / (self.lam_p ** 4)
        
        # MAP Eq. 7 (Paper): These are the components \bar{\alpha}^+ and \bar{\alpha}^-[Paper] 
        # equivalent to the terms (\Theta_c^-(1,0,0) - 1) and (\Theta^+(1) - 1) of SM.B.1[SM].
        self.comp_m = -1.0 / (self.lam_m + 1.0)
        self.comp_p = 1.0 / (self.lam_p - 1.0)

        # Derivatives of Order 1, 2, 3 [SM]
        # We use that the derivatves of thetas evaluated at 0 are 
        # the moments of the exponential distribution.
        # \Theta is the moment generating function (MGF)
        self.dTc_dq0 = self.E_z_m;  self.dTp_dq0 = self.E_z_p
        self.dTc_dq1 = self.mu1 * self.E_z2_m
        self.dTc_dq3 = (1.0 - self.rho3) * self.mu3 * self.E_z2_m
        self.dTi_dq3 = self.rho3 * self.mu3 * self.E_z2_m
        
        self.dTc_dq0_2 = self.E_z2_m; self.dTp_dq0_2 = self.E_z2_p
        self.dTc_dq0_q1 = self.mu1 * self.E_z3_m
        self.dTc_dq0_q3 = (1.0 - self.rho3) * self.mu3 * self.E_z3_m
        
        self.dTc_dq1_2 = (self.mu1 ** 2) * self.E_z4_m
        self.dTc_dq1_q3 = self.mu1 * (1.0 - self.rho3) * self.mu3 * self.E_z4_m
        self.dTc_dq3_2 = ((1.0 - self.rho3) * self.mu3) ** 2 * self.E_z4_m
        self.dTi_dq3_2 = (self.rho3 * self.mu3) ** 2 * self.E_z4_m
        
        self.dTc_dq0_3 = self.E_z3_m; self.dTp_dq0_3 = self.E_z3_p
        self.dTc_dq0_2_q1 = self.mu1 * self.E_z4_m
        self.dTc_dq0_2_q3 = (1.0 - self.rho3) * self.mu3 * self.E_z4_m
        self.dTc_dq0_q1_2 = (self.mu1 ** 2) * self.E_z5_m
        self.dTc_dq0_q1_q3 = self.mu1 * (1.0 - self.rho3) * self.mu3 * self.E_z5_m
        self.dTc_dq0_q3_2 = ((1.0 - self.rho3) * self.mu3) ** 2 * self.E_z5_m
        
        # 4th Derivatives[SM]
        self.dTc_dq0_4 = self.E_z4_m
        self.dTp_dq0_4 = self.E_z4_p
        self.dTc_dq0_3_q1 = self.mu1 * self.E_z5_m
        self.dTc_dq0_3_q3 = (1.0 - self.rho3) * self.mu3 * self.E_z5_m
        self.dTc_dq0_2_q1_2 = (self.mu1 ** 2) * self.E_z6_m
        self.dTc_dq0_2_q1_q3 = self.mu1 * (1.0 - self.rho3) * self.mu3 * self.E_z6_m
        self.dTc_dq0_2_q3_2 = ((1.0 - self.rho3) * self.mu3) ** 2 * self.E_z6_m
        self.dTc_dq0_q1_3 = (self.mu1 ** 3) * self.E_z7_m
        self.dTc_dq0_q1_2_q3 = (self.mu1 ** 2) * (1.0 - self.rho3) * self.mu3 * self.E_z7_m
        self.dTc_dq0_q1_q3_2 = self.mu1 * ((1.0 - self.rho3) * self.mu3) ** 2 * self.E_z7_m
        self.dTc_dq0_q3_3 = ((1.0 - self.rho3) * self.mu3) ** 3 * self.E_z7_m
        self.dTc_dq1_3 = (self.mu1 ** 3) * self.E_z6_m
        self.dTc_dq1_2_q3 = (self.mu1 ** 2) * (1.0 - self.rho3) * self.mu3 * self.E_z6_m
        self.dTc_dq1_q3_2 = self.mu1 * ((1.0 - self.rho3) * self.mu3) ** 2 * self.E_z6_m
        self.dTc_dq3_3 = ((1.0 - self.rho3) * self.mu3) ** 3 * self.E_z6_m
        self.dTi_dq3_3 = (self.rho3 * self.mu3) ** 3 * self.E_z6_m
        self.dTc_dq1_4 = (self.mu1 ** 4) * self.E_z8_m
        self.dTc_dq1_3_q3 = (self.mu1 ** 3) * (1.0 - self.rho3) * self.mu3 * self.E_z8_m
        self.dTc_dq1_2_q3_2 = (self.mu1 ** 2) * ((1.0 - self.rho3) * self.mu3) ** 2 * self.E_z8_m
        self.dTc_dq1_q3_3 = self.mu1 * ((1.0 - self.rho3) * self.mu3) ** 3 * self.E_z8_m
        self.dTc_dq3_4 = ((1.0 - self.rho3) * self.mu3) ** 4 * self.E_z8_m
        self.dTi_dq3_4 = (self.rho3 * self.mu3) ** 4 * self.E_z8_m

    def _get_matrix_A(self):
        # MAP SM.B.1: Represents the A matrix in the ODE[SM].
        # Columns and row orders is [V1, V3, V2].
        A = np.zeros((3, 3))
        A[0, 0] = -self.k1 + self.c1_m * self.dTc_dq1
        A[0, 1] = self.c1_m * (self.dTc_dq3 + self.dTi_dq3)
        A[1, 0] = self.c3_m * self.dTc_dq1
        A[1, 1] = -self.k3 + self.c3_m * (self.dTc_dq3 + self.dTi_dq3)
        A[2, 0] = self.c2_m * self.dTc_dq1
        A[2, 1] = self.c2_m * (self.dTc_dq3 + self.dTi_dq3)
        A[2, 2] = -self.k2
        return A

    def _get_vector_B1(self):
        # MAP SM.B.1: Vector B, that is the partial derivative of the ODE wrt u,
        # evaluated at u=0[SM]. Ordering follows [V1, V3, V2].
        B = np.zeros(3)
        B[0] = -0.5 - self.c1_m * self.comp_m - self.c1_p * self.comp_p + self.c1_m * self.dTc_dq0 + self.c1_p * self.dTp_dq0
        B[1] = -0.5 * (self.eta ** 2) - self.c3_m * self.comp_m - self.c3_p * self.comp_p + self.c3_m * self.dTc_dq0 + self.c3_p * self.dTp_dq0
        B[2] = -0.5 - self.c2_m * self.comp_m - self.c2_p * self.comp_p + self.c2_m * self.dTc_dq0 + self.c2_p * self.dTp_dq0
        return B

    def _precompute_ode_matrices(self):
        # MAP SM.B.2, SM.B.3, SM.B.4: Static matrices: they are 
        # the tensors defined in Section B of the supplemental material,  
        # they are necessary to compute B^(2)(tau), B^(3)(tau) and B^(4)(tau)[SM].
        # B2_c, for example, maps B_{j,c}^{(2)}
        # B2_s maps B_{j,s}^{(2)}, and so on[SM].

        # --- B2 Matrices ---
        self.B2_c = np.array([
            1.0 + self.c1_m * self.dTc_dq0_2 + self.c1_p * self.dTp_dq0_2,
            self.eta**2 + self.c3_m * self.dTc_dq0_2 + self.c3_p * self.dTp_dq0_2,
            1.0 + self.c2_m * self.dTc_dq0_2 + self.c2_p * self.dTp_dq0_2
        ])
        self.B2_s = np.vstack((
            np.array([2*self.sig1*self.rho1 + 2*self.c1_m*self.dTc_dq0_q1, 2*self.c1_m*self.dTc_dq0_q3, 0.0]),
            np.array([2*self.c3_m*self.dTc_dq0_q1, 2*self.c3_m*self.dTc_dq0_q3, 0.0]),
            np.array([2*self.c2_m*self.dTc_dq0_q1, 2*self.c2_m*self.dTc_dq0_q3, 2*self.sig2*self.rho2])
        ))
        self.B2_q_1 = np.array([[self.sig1**2 + self.c1_m*self.dTc_dq1_2, self.c1_m*self.dTc_dq1_q3, 0],
                                [self.c1_m*self.dTc_dq1_q3, self.c1_m*(self.dTc_dq3_2 + self.dTi_dq3_2), 0], [0, 0, 0]])
        self.B2_q_3 = np.array([[self.c3_m*self.dTc_dq1_2, self.c3_m*self.dTc_dq1_q3, 0],
                                [self.c3_m*self.dTc_dq1_q3, self.c3_m*(self.dTc_dq3_2 + self.dTi_dq3_2), 0], [0, 0, 0]])
        self.B2_q_2 = np.array([[self.c2_m*self.dTc_dq1_2, self.c2_m*self.dTc_dq1_q3, 0],
                                [self.c2_m*self.dTc_dq1_q3, self.c2_m*(self.dTc_dq3_2 + self.dTi_dq3_2), 0], [0, 0, self.sig2**2]])

        # SM.B.2: alpha^(2)(tau)
        self.alpha_c_2 = self.c0_m * self.dTc_dq0_2 + self.c0_p * self.dTp_dq0_2
        
        self.alpha_s_2 = np.array([
            2 * self.c0_m * self.dTc_dq0_q1, 
            2 * self.c0_m * self.dTc_dq0_q3, 
            0.0
        ])
        
        self.alpha_q_2 = np.array([
            [self.c0_m * self.dTc_dq1_2, self.c0_m * self.dTc_dq1_q3, 0.0],
            [self.c0_m * self.dTc_dq1_q3, self.c0_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ])


        # --- B3 Matrices ---
        #self.B3_q_s1_2 = self.B3_q_s2_2 = self.B3_q_s3_2 = [0, 0, 0]
        self.B3_c = np.array([
            self.c1_m * self.dTc_dq0_3 + self.c1_p * self.dTp_dq0_3,
            self.c3_m * self.dTc_dq0_3 + self.c3_p * self.dTp_dq0_3,
            self.c2_m * self.dTc_dq0_3 + self.c2_p * self.dTp_dq0_3
        ]) # B_{_,c}^(3) where _ = 1, 3, 2
        self.B3_s = np.vstack((
            np.array([3*self.c1_m*self.dTc_dq0_2_q1, 3*self.c1_m*self.dTc_dq0_2_q3, 0.0]),
            np.array([3*self.c3_m*self.dTc_dq0_2_q1, 3*self.c3_m*self.dTc_dq0_2_q3, 0.0]),
            np.array([3*self.c2_m*self.dTc_dq0_2_q1, 3*self.c2_m*self.dTc_dq0_2_q3, 0.0])
        )) # B_{_,s}^(3) where _ = 1, 3, 2
        self.B3_q_c_1 = np.array([[3*self.c1_m*self.dTc_dq0_q1_2, 3*self.c1_m*self.dTc_dq0_q1_q3, 0],
                                [3*self.c1_m*self.dTc_dq0_q1_q3, 3*self.c1_m*self.dTc_dq0_q3_2, 0], 
                                [0, 0, 0]]) # B_{1, q, c}
        self.B3_q_c_3 = np.array([[3*self.c3_m*self.dTc_dq0_q1_2, 3*self.c3_m*self.dTc_dq0_q1_q3, 0],
                                [3*self.c3_m*self.dTc_dq0_q1_q3, 3*self.c3_m*self.dTc_dq0_q3_2, 0], 
                                [0, 0, 0]]) # B_{3, q, c}
        self.B3_q_c_2 = np.array([[3*self.c2_m*self.dTc_dq0_q1_2, 3*self.c2_m*self.dTc_dq0_q1_q3, 0],
                                [3*self.c2_m*self.dTc_dq0_q1_q3, 3*self.c2_m*self.dTc_dq0_q3_2, 0], 
                                [0, 0, 0]]) # B_{2, q, c}
        # the first 6 variables following this comment are not parameters of "fast_ode_system"
        self.B3_q_s1_1 = np.array([
            self.c1_m * self.dTc_dq1_3, 3 * self.c1_m * self.dTc_dq1_2_q3, 0.0]) # B_{1,q,s}^(1)
        self.B3_q_s1_3 = np.array([
            3 * self.c1_m * self.dTc_dq1_q3_2, self.c1_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 0.0]) # B_{1,q,s}^(3)
        self.B3_q_s3_1 = np.array([
            self.c3_m * self.dTc_dq1_3, 3 * self.c3_m * self.dTc_dq1_2_q3, 0.0]) # B_{3,q,s}^(1)
        self.B3_q_s3_3 = np.array([
            3 * self.c3_m * self.dTc_dq1_q3_2, self.c3_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 0.0]) # B_{3,q,s}^(3)
        self.B3_q_s2_1 = np.array([
            self.c2_m * self.dTc_dq1_3, 3 * self.c2_m * self.dTc_dq1_2_q3, 0.0]) # B_{2, q, s}^(1)
        self.B3_q_s2_3 = np.array([
            3 * self.c2_m * self.dTc_dq1_q3_2, self.c2_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 0.0]) # B_{2, q, s}^(3)
        
        self.B3_q_s_1 = np.vstack((self.B3_q_s1_1, self.B3_q_s3_1, self.B3_q_s2_1))
        self.B3_q_s_3 = np.vstack((self.B3_q_s1_3, self.B3_q_s3_3, self.B3_q_s2_3))
        
        self.B3_h_c = np.vstack((
            np.array([3*self.sig1*self.rho1 + 3*self.c1_m*self.dTc_dq0_q1, 3*self.c1_m*self.dTc_dq0_q3, 0]), #B_{1,h,c}
            np.array([3*self.c3_m*self.dTc_dq0_q1, 3*self.c3_m*self.dTc_dq0_q3, 0]), #B_{3,h,c}
            np.array([3*self.c2_m*self.dTc_dq0_q1, 3*self.c2_m*self.dTc_dq0_q3, 3*self.sig2*self.rho2]) #B_{2,h,c}
        ))
        self.B3_h_s1 = np.array([[3*self.sig1**2 + 3*self.c1_m*self.dTc_dq1_2, 3*self.c1_m*self.dTc_dq1_q3, 0],
                                 [3*self.c1_m*self.dTc_dq1_q3, 3*self.c1_m*(self.dTc_dq3_2 + self.dTi_dq3_2), 0], 
                                 [0, 0, 0]]) #B_{1,h,s}
        self.B3_h_s3 = np.array([[3*self.c3_m*self.dTc_dq1_2, 3*self.c3_m*self.dTc_dq1_q3, 0],
                                 [3*self.c3_m*self.dTc_dq1_q3, 3*self.c3_m*(self.dTc_dq3_2 + self.dTi_dq3_2), 0], 
                                 [0, 0, 0]]) #B_{3,h,s}
        self.B3_h_s2 = np.array([[3*self.c2_m*self.dTc_dq1_2, 3*self.c2_m*self.dTc_dq1_q3, 0],
                                 [3*self.c2_m*self.dTc_dq1_q3, 3*self.c2_m*(self.dTc_dq3_2 + self.dTi_dq3_2), 0], 
                                 [0, 0, 3*self.sig2**2]]) #B_{1,h,s}

        # 2. SM.B.3 alpha^(3) (Pags. SM-8 e SM-9)
        # self.alpha_q_s2 = [0, 0, 0]
        self.alpha_c_3 = self.c0_m * self.dTc_dq0_3 + self.c0_p * self.dTp_dq0_3 #\alpha_c^(3)
        
        self.alpha_s_3 = np.array([
            3 * self.c0_m * self.dTc_dq0_2_q1, 
            3 * self.c0_m * self.dTc_dq0_2_q3, 
            0.0
        ]) # \alpha_s^(3)
        
        self.alpha_q_c = np.array([
            [3 * self.c0_m * self.dTc_dq0_q1_2, 3 * self.c0_m * self.dTc_dq0_q1_q3, 0.0],
            [3 * self.c0_m * self.dTc_dq0_q1_q3, 3 * self.c0_m * self.dTc_dq0_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ]) #\alpha_{q,c}

        self.alpha_q_s_1 = np.array([
            self.c0_m * self.dTc_dq1_3, 3 * self.c0_m * self.dTc_dq1_2_q3, 0.0]) # \alpha_{q, s}^(1)
        self.alpha_q_s_3 = np.array([
            3 * self.c0_m * self.dTc_dq1_q3_2, self.c0_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 0.0]) # \alpha_{q, s}^(3)
        
        self.alpha_h_c = np.array([
            3 * self.c0_m * self.dTc_dq0_q1, 
            3 * self.c0_m * self.dTc_dq0_q3, 
            0.0
        ]) # \alpha_{h, c}
        
        self.alpha_h_s = np.array([
            [3 * self.c0_m * self.dTc_dq1_2, 3 * self.c0_m * self.dTc_dq1_q3, 0.0],
            [3 * self.c0_m * self.dTc_dq1_q3, 3 * self.c0_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ]) # \alpha_{h, s}


        # --- B4 Matrices ---
        # Page SM-10
        # B(4,2)_{j,q,s} = B(4,2)_{j,q,s,h} = [0, 0, 0]^T

        self.B4_c = np.array([
            self.c1_m * self.dTc_dq0_4 + self.c1_p * self.dTp_dq0_4,
            self.c3_m * self.dTc_dq0_4 + self.c3_p * self.dTp_dq0_4,
            self.c2_m * self.dTc_dq0_4 + self.c2_p * self.dTp_dq0_4
        ])
        
        self.B4_s = np.vstack((
            np.array([4 * self.c1_m * self.dTc_dq0_3_q1, 4 * self.c1_m * self.dTc_dq0_3_q3, 0.0]),
            np.array([4 * self.c3_m * self.dTc_dq0_3_q1, 4 * self.c3_m * self.dTc_dq0_3_q3, 0.0]),
            np.array([4 * self.c2_m * self.dTc_dq0_3_q1, 4 * self.c2_m * self.dTc_dq0_3_q3, 0.0])
        ))

        # B(4)_{j,q,c} for j = 1, 2, 3
        self.B4_q_c_1 = np.array([
            [6 * self.c1_m * self.dTc_dq0_2_q1_2, 6 * self.c1_m * self.dTc_dq0_2_q1_q3, 0.0],
            [6 * self.c1_m * self.dTc_dq0_2_q1_q3, 6 * self.c1_m * self.dTc_dq0_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_q_c_3 = np.array([
            [6 * self.c3_m * self.dTc_dq0_2_q1_2, 6 * self.c3_m * self.dTc_dq0_2_q1_q3, 0.0],
            [6 * self.c3_m * self.dTc_dq0_2_q1_q3, 6 * self.c3_m * self.dTc_dq0_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_q_c_2 = np.array([
            [6 * self.c2_m * self.dTc_dq0_2_q1_2, 6 * self.c2_m * self.dTc_dq0_2_q1_q3, 0.0],
            [6 * self.c2_m * self.dTc_dq0_2_q1_q3, 6 * self.c2_m * self.dTc_dq0_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])

        # B(4,1)_{1,q,s} e B(4,3)_{1,q,s}
        self.B4_q_s_1_1 = np.array([4 * self.c1_m * self.dTc_dq0_q1_3, 12 * self.c1_m * self.dTc_dq0_q1_2_q3, 0.0])
        self.B4_q_s_3_1 = np.array([12 * self.c1_m * self.dTc_dq0_q1_q3_2, 4 * self.c1_m * self.dTc_dq0_q3_3, 0.0])

        # B(4,1)_{3,q,s} e B(4,3)_{3,q,s}
        self.B4_q_s_1_3 = np.array([4 * self.c3_m * self.dTc_dq0_q1_3, 12 * self.c3_m * self.dTc_dq0_q1_2_q3, 0.0])
        self.B4_q_s_3_3 = np.array([12 * self.c3_m * self.dTc_dq0_q1_q3_2, 4 * self.c3_m * self.dTc_dq0_q3_3, 0.0])

        # B(4,1)_{2,q,s} e B(4,3)_{2,q,s}
        self.B4_q_s_1_2 = np.array([4 * self.c2_m * self.dTc_dq0_q1_3, 12 * self.c2_m * self.dTc_dq0_q1_2_q3, 0.0])
        self.B4_q_s_3_2 = np.array([12 * self.c2_m * self.dTc_dq0_q1_q3_2, 4 * self.c2_m * self.dTc_dq0_q3_3, 0.0])

        # Page SM-11
        # B(4,1)_{1,q,q} e B(4,3)_{1,q,q}
        self.B4_q_q_1_1 = np.array([
            [self.c1_m * self.dTc_dq1_4, 2 * self.c1_m * self.dTc_dq1_3_q3, 0.0],
            [2 * self.c1_m * self.dTc_dq1_3_q3, 3 * self.c1_m * self.dTc_dq1_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_q_q_3_1 = np.array([
            [3 * self.c1_m * self.dTc_dq1_2_q3_2, 2 * self.c1_m * self.dTc_dq1_q3_3, 0.0],
            [2 * self.c1_m * self.dTc_dq1_q3_3, self.c1_m * (self.dTc_dq3_4 + self.dTi_dq3_4), 0.0],
            [0.0, 0.0, 0.0]
        ])

        # B(4,1)_{3,q,q} e B(4,3)_{3,q,q}
        self.B4_q_q_1_3 = np.array([
            [self.c3_m * self.dTc_dq1_4, 2 * self.c3_m * self.dTc_dq1_3_q3, 0.0],
            [2 * self.c3_m * self.dTc_dq1_3_q3, 3 * self.c3_m * self.dTc_dq1_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_q_q_3_3 = np.array([
            [3 * self.c3_m * self.dTc_dq1_2_q3_2, 2 * self.c3_m * self.dTc_dq1_q3_3, 0.0],
            [2 * self.c3_m * self.dTc_dq1_q3_3, self.c3_m * (self.dTc_dq3_4 + self.dTi_dq3_4), 0.0],
            [0.0, 0.0, 0.0]
        ])

        # B(4,1)_{2,q,q} e B(4,3)_{2,q,q}
        self.B4_q_q_1_2 = np.array([
            [self.c2_m * self.dTc_dq1_4, 2 * self.c2_m * self.dTc_dq1_3_q3, 0.0],
            [2 * self.c2_m * self.dTc_dq1_3_q3, 3 * self.c2_m * self.dTc_dq1_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_q_q_3_2 = np.array([
            [3 * self.c2_m * self.dTc_dq1_2_q3_2, 2 * self.c2_m * self.dTc_dq1_q3_3, 0.0],
            [2 * self.c2_m * self.dTc_dq1_q3_3, self.c2_m * (self.dTc_dq3_4 + self.dTi_dq3_4), 0.0],
            [0.0, 0.0, 0.0]
        ])

        # We will not use the first 6 variables under this comment
        # B(4,1)_{1,q,s,h} e B(4,3)_{1,q,s,h}
        self.B4_q_s_h_1_1 = np.array([6 * self.c1_m * self.dTc_dq1_3, 6 * self.c1_m * self.dTc_dq1_2_q3, 0.0])
        self.B4_q_s_h_3_1 = np.array([6 * self.c1_m * self.dTc_dq1_q3_2, 6 * self.c1_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 0.0])

        # B(4,1)_{3,q,s,h} e B(4,3)_{3,q,s,h}
        self.B4_q_s_h_1_3 = np.array([6 * self.c3_m * self.dTc_dq1_3, 6 * self.c3_m * self.dTc_dq1_2_q3, 0.0])
        self.B4_q_s_h_3_3 = np.array([6 * self.c3_m * self.dTc_dq1_q3_2, 6 * self.c3_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 0.0])

        # B(4,1)_{2,q,s,h} e B(4,3)_{2,q,s,h}
        self.B4_q_s_h_1_2 = np.array([6 * self.c2_m * self.dTc_dq1_3, 6 * self.c2_m * self.dTc_dq1_2_q3, 0.0])
        self.B4_q_s_h_3_2 = np.array([6 * self.c2_m * self.dTc_dq1_q3_2, 6 * self.c2_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 0.0])

        # We will use this 2 instead
        self.B4_q_s_h_1 = np.vstack((self.B4_q_s_h_1_1, self.B4_q_s_h_1_3, self.B4_q_s_h_1_2))
        self.B4_q_s_h_3 = np.vstack((self.B4_q_s_h_3_1, self.B4_q_s_h_3_3, self.B4_q_s_h_3_2))

        # B(4,1,2)_{j,q,s,h}
        self.B4_q_s_h_12 = np.vstack((
            np.array([6 * self.c1_m * self.dTc_dq1_2_q3, 6 * self.c1_m * self.dTc_dq1_q3_2, 0.0]),
            np.array([6 * self.c3_m * self.dTc_dq1_2_q3, 6 * self.c3_m * self.dTc_dq1_q3_2, 0.0]),
            np.array([6 * self.c2_m * self.dTc_dq1_2_q3, 6 * self.c2_m * self.dTc_dq1_q3_2, 0.0])
        ))

        # B(4)_{j,h,c}
        self.B_j_h_c = np.vstack((
            np.array([6 * self.c1_m * self.dTc_dq0_2_q1, 6 * self.c1_m * self.dTc_dq0_2_q3, 0.0]),
            np.array([6 * self.c3_m * self.dTc_dq0_2_q1, 6 * self.c3_m * self.dTc_dq0_2_q3, 0.0]),
            np.array([6 * self.c2_m * self.dTc_dq0_2_q1, 6 * self.c2_m * self.dTc_dq0_2_q3, 0.0])
        ))

        # B(4)_{1,h,s}, B(4)_{3,h,s}, B(4)_{2,h,s}
        self.B4_h_s_1 = np.array([
            [12 * self.c1_m * self.dTc_dq0_q1_2, 12 * self.c1_m * self.dTc_dq0_q1_q3, 0.0],
            [12 * self.c1_m * self.dTc_dq0_q1_q3, 12 * self.c1_m * self.dTc_dq0_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_h_s_3 = np.array([
            [12 * self.c3_m * self.dTc_dq0_q1_2, 12 * self.c3_m * self.dTc_dq0_q1_q3, 0.0],
            [12 * self.c3_m * self.dTc_dq0_q1_q3, 12 * self.c3_m * self.dTc_dq0_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_h_s_2 = np.array([
            [12 * self.c2_m * self.dTc_dq0_q1_2, 12 * self.c2_m * self.dTc_dq0_q1_q3, 0.0],
            [12 * self.c2_m * self.dTc_dq0_q1_q3, 12 * self.c2_m * self.dTc_dq0_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])

        # Page SM-12
        # B(4)_{1,l,c}, B(4)_{3,l,c}, B(4)_{2,l,c}
        self.B4_l_c = np.vstack((
            np.array([4 * self.sig1 * self.rho1 + 4 * self.c1_m * self.dTc_dq0_q1, 4 * self.c1_m * self.dTc_dq0_q3, 0.0]),
            np.array([4 * self.c3_m * self.dTc_dq0_q1, 4 * self.c3_m * self.dTc_dq0_q3, 0.0]),
            np.array([4 * self.c2_m * self.dTc_dq0_q1, 4 * self.c2_m * self.dTc_dq0_q3, 4 * self.sig2 * self.rho2])
        ))

        # B(4)_{1,l,s}, B(4)_{3,l,s}, B(4)_{2,l,s}
        self.B4_l_s_1 = np.array([
            [4 * self.sig1**2 + 4 * self.c1_m * self.dTc_dq1_2, 4 * self.c1_m * self.dTc_dq1_q3, 0.0],
            [4 * self.c1_m * self.dTc_dq1_q3, 4 * self.c1_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_l_s_3 = np.array([
            [4 * self.c3_m * self.dTc_dq1_2, 4 * self.c3_m * self.dTc_dq1_q3, 0.0],
            [4 * self.c3_m * self.dTc_dq1_q3, 4 * self.c3_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_l_s_2 = np.array([
            [4 * self.c2_m * self.dTc_dq1_2, 4 * self.c2_m * self.dTc_dq1_q3, 0.0], # Corretto il typo dq2_1 del paper in dq1_2
            [4 * self.c2_m * self.dTc_dq1_q3, 4 * self.c2_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0], # Aggiunta c2_m in dTi_dq3_2
            [0.0, 0.0, 4 * self.sig2**2]
        ])

        # B(4)_{1,h,q}, B(4)_{3,h,q}, B(4)_{2,h,q}
        self.B4_h_q_1 = np.array([
            [3 * self.sig1**2 + 3 * self.c1_m * self.dTc_dq1_2, 3 * self.c1_m * self.dTc_dq1_q3, 0.0],
            [3 * self.c1_m * self.dTc_dq1_q3, 3 * self.c1_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_h_q_3 = np.array([
            [3 * self.c3_m * self.dTc_dq1_2, 3 * self.c3_m * self.dTc_dq1_q3, 0.0],
            [3 * self.c3_m * self.dTc_dq1_q3, 3 * self.c3_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.B4_h_q_2 = np.array([
            [3 * self.c2_m * self.dTc_dq1_2, 3 * self.c2_m * self.dTc_dq1_q3, 0.0],
            [3 * self.c2_m * self.dTc_dq1_q3, 3 * self.c2_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 3 * self.sig2**2]
        ])

        # SM.B.4 alpha^(4) (Pags. SM-14 e SM-15)
        self.alpha_c_4 = self.c0_m * self.dTc_dq0_4 + self.c0_p * self.dTp_dq0_4
        
        self.alpha_s_4 = np.array([
            4 * self.c0_m * self.dTc_dq0_3_q1, 
            4 * self.c0_m * self.dTc_dq0_3_q3, 
            0.0
        ])

        self.alpha_q_c_4 = np.array([
            [6 * self.c0_m * self.dTc_dq0_2_q1_2, 6 * self.c0_m * self.dTc_dq0_2_q1_q3, 0.0],
            [6 * self.c0_m * self.dTc_dq0_2_q1_q3, 6 * self.c0_m * self.dTc_dq0_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])

        self.alpha_q_s_4_1 = np.array([
            4 * self.c0_m * self.dTc_dq0_q1_3, 
            12 * self.c0_m * self.dTc_dq0_q1_2_q3, 
            0.0
        ])
        self.alpha_q_s_4_3 = np.array([
            12 * self.c0_m * self.dTc_dq0_q1_q3_2, 
            4 * self.c0_m * self.dTc_dq0_q3_3, 
            0.0
        ])

        self.alpha_q_q_4_1 = np.array([
            [self.c0_m * self.dTc_dq1_4, 2 * self.c0_m * self.dTc_dq1_3_q3, 0.0],
            [2 * self.c0_m * self.dTc_dq1_3_q3, 3 * self.c0_m * self.dTc_dq1_2_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.alpha_q_q_4_3 = np.array([
            [3 * self.c0_m * self.dTc_dq1_2_q3_2, 2 * self.c0_m * self.dTc_dq1_q3_3, 0.0],
            [2 * self.c0_m * self.dTc_dq1_q3_3, self.c0_m * (self.dTc_dq3_4 + self.dTi_dq3_4), 0.0],
            [0.0, 0.0, 0.0]
        ])

        self.alpha_q_s_h_4_1 = np.array([
            6 * self.c0_m * self.dTc_dq1_3, 
            6 * self.c0_m * self.dTc_dq1_2_q3, 
            0.0
        ])
        self.alpha_q_s_h_4_3 = np.array([
            6 * self.c0_m * self.dTc_dq1_q3_2, 
            6 * self.c0_m * (self.dTc_dq3_3 + self.dTi_dq3_3), 
            0.0
        ])

        # alpha_{q,s,h}^(4,1,2)
        self.alpha_q_s_h_4_1_2 = np.array([
            6 * self.c0_m * self.dTc_dq1_2_q3, 
            6 * self.c0_m * self.dTc_dq1_q3_2, 
            0.0
        ])

        # Page SM-15
        self.alpha_h_c_4 = np.array([
            6 * self.c0_m * self.dTc_dq0_2_q1, 
            6 * self.c0_m * self.dTc_dq0_2_q3, 
            0.0
        ])
        
        self.alpha_h_s_4 = np.array([
            [12 * self.c0_m * self.dTc_dq0_q1_2, 12 * self.c0_m * self.dTc_dq0_q1_q3, 0.0],
            [12 * self.c0_m * self.dTc_dq0_q1_q3, 12 * self.c0_m * self.dTc_dq0_q3_2, 0.0],
            [0.0, 0.0, 0.0]
        ])

        self.alpha_l_c_4 = np.array([
            4 * self.c0_m * self.dTc_dq0_q1, 
            4 * self.c0_m * self.dTc_dq0_q3, 
            0.0
        ])
        
        self.alpha_l_s_4 = np.array([
            [4 * self.c0_m * self.dTc_dq1_2, 4 * self.c0_m * self.dTc_dq1_q3, 0.0],
            [4 * self.c0_m * self.dTc_dq1_q3, 4 * self.c0_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ])

        self.alpha_h_q_4 = np.array([
            [3 * self.c0_m * self.dTc_dq1_2, 3 * self.c0_m * self.dTc_dq1_q3, 0.0],
            [3 * self.c0_m * self.dTc_dq1_q3, 3 * self.c0_m * (self.dTc_dq3_2 + self.dTi_dq3_2), 0.0],
            [0.0, 0.0, 0.0]
        ])

    def get_cumulant_loadings(self, tau):
        # MAP Eq. 6 (Paper): Output of this function \Gamma_0^(j) and
        # the components of the loading factor \Gamma_1^(j)[Paper] to obtain the cumulant
        # for a specific expiring time (tau) integrating them on ODE.
        Y0 = np.zeros(15) 

        # Pre-building of 3D array for the cumulants (To speed up Numba)
        self.B2_q = np.array([self.B2_q_1, self.B2_q_3, self.B2_q_2])
        
        self.B3_q_c = np.array([self.B3_q_c_1, self.B3_q_c_3, self.B3_q_c_2])
        self.B3_h_s = np.array([self.B3_h_s1, self.B3_h_s3, self.B3_h_s2])

        B4_q_c = np.array([self.B4_q_c_1, self.B4_q_c_3, self.B4_q_c_2])
        B4_q_s_1_4 = np.vstack((self.B4_q_s_1_1, self.B4_q_s_1_3, self.B4_q_s_1_2))
        B4_q_s_3_4 = np.vstack((self.B4_q_s_3_1, self.B4_q_s_3_3, self.B4_q_s_3_2))
        B4_q_q_1 = np.array([self.B4_q_q_1_1, self.B4_q_q_1_3, self.B4_q_q_1_2])
        B4_q_q_3 = np.array([self.B4_q_q_3_1, self.B4_q_q_3_3, self.B4_q_q_3_2])
        B4_h_s = np.array([self.B4_h_s_1, self.B4_h_s_3, self.B4_h_s_2])
        B4_h_q = np.array([self.B4_h_q_1, self.B4_h_q_3, self.B4_h_q_2])
        B4_l_s = np.array([self.B4_l_s_1, self.B4_l_s_3, self.B4_l_s_2]) #
        B4_q_s_h_1 = np.vstack((self.B4_q_s_h_1_1, self.B4_q_s_h_1_3, self.B4_q_s_h_1_2))
        B4_q_s_h_3 = np.vstack((self.B4_q_s_h_3_1, self.B4_q_s_h_3_3, self.B4_q_s_h_3_2))
        B4_h_s = np.array([self.B4_h_s_1, self.B4_h_s_3, self.B4_h_s_2])
        B4_h_q = np.array([self.B4_h_q_1, self.B4_h_q_3, self.B4_h_q_2])
        B4_l_s = np.array([self.B4_l_s_1, self.B4_l_s_3, self.B4_l_s_2])
        
        # ===========================================================
        # RUNGE-KUTTA 4 CALL
        # ===========================================================
        # Dynamic computing of the number of steps: 
        # Set 5 micro-step for each day.
        # For tau = 1 month (21/252) -> ~ 105 steps
        # For tau = 1 year (252/252) -> ~ 1260 steps
        num_steps = max(int(np.ceil(tau * 252.0 * 5.0)), 10)

        Y_final = rk4_integrator(
            tau, Y0, num_steps,
            self.A, self.B1, self.A_alpha, 
            self.B2_c, self.B2_s, self.B2_q,  
            self.alpha_c_2, self.alpha_s_2, self.alpha_q_2, 
            self.B3_c, self.B3_s, self.B3_q_c, self.B3_q_s_1, self.B3_q_s_3,
            self.B3_h_c, self.B3_h_s, 
            self.alpha_c_3, self.alpha_s_3, self.alpha_q_c, self.alpha_q_s_1, self.alpha_q_s_3,
            self.alpha_h_c, self.alpha_h_s, 
            self.B4_c, self.B4_s, B4_q_c, B4_q_s_1_4, B4_q_s_3_4, B4_q_q_1, B4_q_q_3, 
            B4_q_s_h_1, B4_q_s_h_3, self.B4_q_s_h_12, 
            self.B_j_h_c, B4_h_s, B4_h_q, 
            self.B4_l_c, B4_l_s, 
            self.alpha_c_4, self.alpha_s_4, self.alpha_q_c_4, self.alpha_q_s_4_1, self.alpha_q_s_4_3, self.alpha_q_q_4_1, self.alpha_q_q_4_3,
            self.alpha_q_s_h_4_1, self.alpha_q_s_h_4_3, self.alpha_q_s_h_4_1_2,
            self.alpha_h_c_4, self.alpha_h_s_4, self.alpha_h_q_4,
            self.alpha_l_c_4, self.alpha_l_s_4
        )
        
        b1, b2, b3, b4 = Y_final[0:3], Y_final[3:6], Y_final[6:9], Y_final[9:12]
        a2, a3, a4 = Y_final[12], Y_final[13], Y_final[14]
        
        # MAP SM.B: standard reodering: \beta variable in [SM] 
        # follows the order [V1, V3, V2] due to jumps dependency [SM].
        # Here \beta follows [V1, V2, V3] [Paper] (Eq. 6)
        # to compute easily the final matricial product [Paper, SM].
        g1 = np.array([b1[0], b1[2], b1[1]]).reshape(3, 1)
        g2 = np.array([b2[0], b2[2], b2[1]]).reshape(3, 1)
        g3 = np.array([b3[0], b3[2], b3[1]]).reshape(3, 1)
        g4 = np.array([b4[0], b4[2], b4[1]]).reshape(3, 1)
        
        g0_2 = np.array([[a2]])
        g0_3 = np.array([[a3]])
        g0_4 = np.array([[a4]])
        
        return g1, g2, g3, g4, g0_2, g0_3, g0_4

# Trial
if __name__ == "__main__":
    theta_test = {
        'k1': 0.62, 'vbar1': 0.026, 'sig1': 0.17, 'rho1': -0.98,
        'k2': 0.70, 'vbar2': 0.019, 'sig2': 0.14, 'rho2': -0.98,
        'k3': 13.7, 'mu3': 1.36, 'rho3': 0.73,
        'mu1': 0.08, 'lam_m': 13.96, 'lam_p': 248.4,
        'c0_m': 0.0, 'c1_m': 55.49, 'c2_m': 1.64, 'c3_m': 451.7,
        'c0_p': 4.55, 'c1_p': 256.4, 'c2_p': 212.4, 'c3_p': 0.0,
        'eta': 0.0, 'r': 0.0, 'delta': 0.0
    }
    theta_val = {
        'k1': 0.5, 'vbar1': 0.02, 'sig1': 0.1, 'rho1': -0.9800,
        'k2': 0.3, 'vbar2': 0.04, 'sig2': 0.15, 'rho2': -0.9897, 
        'k3': 50.0,'mu3': 0.6, 'rho3': 0.06, 
        'mu1': 0.2, 'lam_m': 7.0, 'lam_p': 240.0, 
        'c0_m': 0.0, 'c1_m': 20.0, 'c2_m': 0.1, 'c3_m': 160.0, 
        'c0_p': 5.0, 'c1_p': 245.0, 'c2_p': 100.0, 'c3_p': 0.0, 
        'eta': 0.0, 'r': 0.0, 'delta': 0.0}
    
    engine = AFTCumulantCalculator(theta_val)
    tau_1_month = 21.0 / 252.0 
    
    g1, g2, g3, g4, g0_2, g0_3, g0_4 = engine.get_cumulant_loadings(1)
    
    print("KURTOSIS (4° Cumulant):")
    print(f"Gamma_0: {g0_4[0,0]:.6f}")
    print(f"Gamma_1:\n{g4}")