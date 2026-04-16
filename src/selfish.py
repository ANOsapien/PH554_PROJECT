import numpy as np
from numba import njit

PI2 = 2.0 * np.pi

# ----------------------------
# Initialization (keep in Python)
# ----------------------------

def initialize_lattice(L, mode="ordered"):
    if mode == "ordered":
        return np.zeros((L, L))
    elif mode == "disordered":
        return np.random.uniform(0, 2*np.pi, (L, L))
    else:
        raise ValueError("Unknown mode")


# ----------------------------
# Fast angle difference (inlined)
# ----------------------------

@njit
def angle_diff_scalar(a, b):
    d = abs(a - b)
    if d > np.pi:
        return PI2 - d
    return d


# ----------------------------
# Single-site update (FAST)
# ----------------------------

@njit
def glauber_update_numba(phi, beta, cone_half, Delta):
    """
    cone_half : half-angle of vision cone in radians  = (full_cone_deg / 2) in rad
                Equivalent to Psi in auxiliary.py and theta/2 in Bandini paper.
    Delta     : max perturbation size (same as Delta in auxiliary.py, default 0.5).

    FIX 1: Proposal is now a SMALL PERTURBATION old_phi + uniform(-Delta, Delta),
    NOT a full resample.  Both Bandini et al. App. A and Shi et al. Sec. C2a
    require a small-delta Glauber update for the dynamics to be valid.
    Previous code used:  new_phi = np.random.rand() * PI2   <-- full resample, WRONG.
    """
    L = phi.shape[0]

    i = np.random.randint(L)
    j = np.random.randint(L)

    old_phi = phi[i, j]

    # FIX 1: small-step perturbation (was: new_phi = np.random.rand() * PI2)
    delta = (np.random.rand() * 2.0 - 1.0) * Delta
    new_phi = (old_phi + delta) % PI2

    # neighbors (manual, no allocation)
    right = phi[i, (j+1) % L]
    up    = phi[(i+1) % L, j]
    left  = phi[i, (j-1) % L]
    down  = phi[(i-1) % L, j]

    dirs = (0.0, 0.5*np.pi, np.pi, 1.5*np.pi)

    # ---- OLD SELFISH ENERGY (Bandini Eq.2) ----
    # E^NR_i = -sum_j J_ij(phi_i) * cos(phi_i - phi_j)
    # J_ij = 1 if neighbor j is within vision cone of spin i, else 0
    E_old = 0.0

    if angle_diff_scalar(old_phi, dirs[0]) <= cone_half:
        E_old -= np.cos(old_phi - right)
    if angle_diff_scalar(old_phi, dirs[1]) <= cone_half:
        E_old -= np.cos(old_phi - up)
    if angle_diff_scalar(old_phi, dirs[2]) <= cone_half:
        E_old -= np.cos(old_phi - left)
    if angle_diff_scalar(old_phi, dirs[3]) <= cone_half:
        E_old -= np.cos(old_phi - down)

    # ---- NEW SELFISH ENERGY ----
    E_new = 0.0

    if angle_diff_scalar(new_phi, dirs[0]) <= cone_half:
        E_new -= np.cos(new_phi - right)
    if angle_diff_scalar(new_phi, dirs[1]) <= cone_half:
        E_new -= np.cos(new_phi - up)
    if angle_diff_scalar(new_phi, dirs[2]) <= cone_half:
        E_new -= np.cos(new_phi - left)
    if angle_diff_scalar(new_phi, dirs[3]) <= cone_half:
        E_new -= np.cos(new_phi - down)

    dE = E_new - E_old

    # Glauber probability (Bandini Eq.1, Shi Eq.6)
    # w = 0.5 * (1 - tanh(beta * dE / 2))
    w = 0.5 * (1.0 - np.tanh(beta * dE * 0.5))

    if np.random.rand() < w:
        phi[i, j] = new_phi
        return dE, 1
    else:
        return dE, 0


# ----------------------------
# Magnetization (FAST)
# ----------------------------

@njit
def magnetization_numba(phi):
    L = phi.shape[0]

    sumx = 0.0
    sumy = 0.0

    for i in range(L):
        for j in range(L):
            sumx += np.cos(phi[i, j])
            sumy += np.sin(phi[i, j])

    N = L * L
    mx = sumx / N
    my = sumy / N
    m = np.sqrt(mx*mx + my*my)

    return m, mx, my


# ----------------------------
# FULL SIMULATION (NUMBA)
# ----------------------------

@njit
def run_simulation_numba(phi, beta, cone_half, Delta, steps, thermal_steps, sample_gap):
    """
    cone_half  : Psi = half-angle of vision cone in radians
    Delta      : perturbation size for Glauber proposal
    sample_gap : measure every sample_gap MC steps  (set = L*L to match auxiliary.py)
    """
    L = phi.shape[0]
    N = L * L

    max_samples = steps // sample_gap + 1

    mx_vals = np.zeros(max_samples)
    my_vals = np.zeros(max_samples)

    m_sum = 0.0
    epr_sum = 0.0

    sample_count = 0
    epr_count = 0

    for step in range(steps):

        dE, accepted = glauber_update_numba(phi, beta, cone_half, Delta)

        # EPR: log-ratio of reverse to forward transition rates (Bandini Eq.21, Shi Eq.6)
        # w_f = 0.5*(1 - tanh(beta*dE/2))
        # w_r = 0.5*(1 - tanh(-beta*dE/2)) = 0.5*(1 + tanh(beta*dE/2))
        # log(w_r/w_f) is nonzero only when accepted (detailed balance violation)
        if accepted == 1:
            w_f = 0.5 * (1.0 - np.tanh(beta * dE * 0.5))
            w_r = 0.5 * (1.0 - np.tanh(-beta * dE * 0.5))
            epr_sum += np.log(w_r / w_f)
            epr_count += 1

        if step > thermal_steps and step % sample_gap == 0:
            m, mx, my = magnetization_numba(phi)

            m_sum += m
            mx_vals[sample_count] = mx
            my_vals[sample_count] = my
            sample_count += 1

    return m_sum / sample_count, mx_vals[:sample_count], my_vals[:sample_count], \
           (epr_sum / epr_count if epr_count > 0 else 0.0)


# ----------------------------
# Python wrapper
# ----------------------------

def run_simulation(L=64, T=1.0, theta_deg=180,
                   steps=None, init="disordered"):
    """
    theta_deg : full vision cone angle in degrees.
                Internally converted to cone_half = deg2rad(theta_deg) / 2,
                which equals Psi in auxiliary.py and Shi et al.
                This matches Bandini et al.'s J_ij check: |phi_i - vartheta_ij| <= theta/2.
    """
    beta = 1.0 / T

    # FIX 5 (notation): renamed internal variable from 'theta' to 'cone_half'
    # to avoid confusion with the spin-angle variable 'theta' used in auxiliary.py.
    # cone_half == Psi in Shi et al. == theta/2 in Bandini et al.
    cone_half = np.deg2rad(theta_deg) / 2.0

    # Delta: perturbation size, set equal to auxiliary.py's Delta = 0.5
    Delta = 0.5

    phi = initialize_lattice(L, init)

    if steps is None:
        steps = int(20 * L ** 4)

    thermal_steps = int(0.3 * steps)

    # FIX 3 (matched to auxiliary.py): sample_gap = L*L = N
    # Old code had sample_gap = L*L already, but auxiliary.py used N*5.
    # Both are now set to N (= L*L).
    sample_gap = L * L

    m, mx, my, epr = run_simulation_numba(
        phi, beta, cone_half, Delta, steps, thermal_steps, sample_gap
    )

    return {
        "m": m,
        "mx": mx,
        "my": my,
        "EPR": epr
    }