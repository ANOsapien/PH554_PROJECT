import numpy as np
from numba import njit
import math

# =========================
# NEIGHBOR STRUCTURE
# =========================
# psi_dir: lattice directions psi_ij in Shi et al. = vartheta_ij in Bandini et al.
# These are the angles of the four nearest-neighbor bond directions on the square lattice.
dx = np.array([1, 0, -1, 0])
dy = np.array([0, 1, 0, -1])
psi_dir = np.array([0, np.pi/2, np.pi, 3*np.pi/2])


# =========================
# HELPERS
# =========================

@njit
def angle_diff(a, b):
    """Signed angle difference, wrapped to (-pi, pi]."""
    d = a - b
    while d > np.pi:
        d -= 2*np.pi
    while d < -np.pi:
        d += 2*np.pi
    return d


@njit
def in_cone(theta_i, direction, Psi):
    """
    Returns True if lattice direction 'direction' is within the vision cone of spin theta_i.

    Notation correspondence:
      theta_i   : spin angle (Shi: theta_i,  Bandini: phi_i)
      direction : bond direction psi_ij  (Shi) = vartheta_ij (Bandini)
      Psi       : half-angle of vision cone (Shi: Psi) = theta/2 (Bandini)

    Coupling J_ij(theta_i) = 1 iff in_cone(theta_i, psi_ij, Psi) is True.
    This matches both Shi Eq.(main text) and Bandini Eq.(3).
    """
    return abs(angle_diff(direction, theta_i)) <= Psi


# =========================
# LOCAL ΔH  (Shi et al. constrained Hamiltonian, Eq. 2 / A10)
# =========================
# H = H_SS + H_Sa
# H_SS = -sum_{<ij>} [J_ij(theta_i) + J_ji(theta_j)] cos(theta_i - theta_j)
# H_Sa = -sum_{<ij>} [J_ij(theta_i) cos(theta_j - phi_i) + J_ji(theta_j) cos(theta_i - phi_j)]
#
# Under the mirror constraint: phi_i = theta_i - pi  =>  phi_i - theta_j = theta_i - pi - theta_j
# The update proposes theta_i -> theta_new while holding phi_i fixed (individual-update scheme,
# Shi Sec. C2a).  phi_i is still the OLD value = old_theta - pi.

@njit
def delta_H(theta, x, y, theta_new, L, Psi):
    """
    Compute ΔH = H(theta_new, phi_old) - H(theta_old, phi_old)
    where phi_old = old_theta - pi  (mirror constraint, held fixed during proposal).

    This is exactly the energy difference used in Shi Eq.(E1)/(E2).
    """
    old_theta = theta[x, y]

    # phi values under constraint: phi_i = theta_i - pi
    phi_old = old_theta - np.pi   # auxiliary spin for site (x,y), held fixed
    phi_new = theta_new - np.pi   # what phi would be AFTER constraint is enforced (not used here)

    dH = 0.0

    for k in range(4):
        nx = (x + dx[k]) % L
        ny = (y + dy[k]) % L

        theta_j = theta[nx, ny]
        phi_j   = theta_j - np.pi   # auxiliary of neighbor, always under constraint

        direction_ij = psi_dir[k]          # bond direction i -> j
        direction_ji = psi_dir[(k + 2) % 4]  # bond direction j -> i

        # Vision-cone couplings
        Jij_old = 1.0 if in_cone(old_theta, direction_ij, Psi) else 0.0
        Jji     = 1.0 if in_cone(theta_j,   direction_ji, Psi) else 0.0  # unchanged by proposal

        Jij_new = 1.0 if in_cone(theta_new, direction_ij, Psi) else 0.0

        # H_SS contribution for bond (i,j):
        # -[J_ij(theta_i) + J_ji(theta_j)] * cos(theta_i - theta_j)
        old_ss = -(Jij_old + Jji) * math.cos(old_theta - theta_j)
        new_ss = -(Jij_new + Jji) * math.cos(theta_new - theta_j)

        # H_Sa contribution for bond (i,j):
        # -J_ij(theta_i)*cos(theta_j - phi_i)  -J_ji(theta_j)*cos(theta_i - phi_j)
        # Only the first term changes with the proposal (phi_i = phi_old is fixed;
        # J_ji and phi_j are both unchanged).
        old_sa = -Jij_old * math.cos(theta_j - phi_old) \
                 -Jji     * math.cos(old_theta - phi_j)

        new_sa = -Jij_new * math.cos(theta_j - phi_old) \
                 -Jji     * math.cos(theta_new - phi_j)

        dH += (new_ss + new_sa) - (old_ss + old_sa)

    return dH


# =========================
# MC STEP + EPR
# =========================

@njit
def mc_step(theta, T, L, Psi, Delta):
    """
    Individual-update constrained Glauber step (Shi Sec. C2a / Algorithm Sec. E1).

    1. Pick random site (x, y).
    2. Propose theta_new = theta_old + uniform(-Delta, Delta).
    3. Compute ΔH = H(theta_new, phi_old) - H(theta_old, phi_old),  phi_old fixed.
    4. Accept with Glauber probability w = 0.5*(1 - tanh(ΔH / (2T))).
    5. If accepted: update theta[x,y] = theta_new.
       (phi is implicitly enforced as theta - pi on the fly; no separate array needed.)

    Delta = 0.5 rad  (same as selfish.py, matching Shi et al. Sec. E1 recommendation
    to use small Delta for best agreement with Langevin dynamics.)
    """
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)

    delta = (np.random.rand() * 2 - 1) * Delta
    theta_new = theta[x, y] + delta

    # wrap to [0, 2pi)
    if theta_new > 2*np.pi:
        theta_new -= 2*np.pi
    elif theta_new < 0:
        theta_new += 2*np.pi

    dH = delta_H(theta, x, y, theta_new, L, Psi)

    # Glauber acceptance (Shi Eq.6 = Bandini Eq.1)
    prob = 0.5 * (1.0 - math.tanh(dH / (2*T)))

    accepted = False
    if np.random.rand() < prob:
        theta[x, y] = theta_new
        accepted = True

    return dH, accepted


# =========================
# OBSERVABLES
# =========================

@njit
def compute_mxy(theta):
    mx = 0.0
    my = 0.0
    L = theta.shape[0]

    for i in range(L):
        for j in range(L):
            mx += math.cos(theta[i, j])
            my += math.sin(theta[i, j])

    mx /= (L*L)
    my /= (L*L)

    return mx, my


# =========================
# MAIN API (MATCH DRIVER)
# =========================

def run_simulation(L, T, theta_deg, init="disordered"):
    """
    theta_deg : full vision cone angle in degrees.
                Internally: Psi = deg2rad(theta_deg) / 2  (half-angle).
                Matches selfish.py's cone_half and Shi et al.'s Psi.
    Delta     : perturbation size = 0.5 rad.
                Matches selfish.py and Shi et al. Sec. E1.
    """
    N = L * L
    Psi   = np.deg2rad(theta_deg) / 2.0
    Delta = 0.5  # same as selfish.py

    burn_steps   = int(0.3 * (20 * L ** 4 ))
    sample_steps = int(0.7 * (20 * L ** 4 ))

    # --- init ---
    if init == "disordered":
        theta = np.random.rand(L, L) * 2 * np.pi
    else:
        theta = np.zeros((L, L))

    # --- burn-in ---
    for _ in range(burn_steps):
        mc_step(theta, T, L, Psi, Delta)

    # --- sampling ---
    mx_acc  = 0.0
    my_acc  = 0.0
    m_acc   = 0.0
    epr_acc = 0.0
    count   = 0

    # FIX 3: sample every N = L*L steps, matching selfish.py's sample_gap = L*L.
    # Old code sampled every N*5 steps (5x sparser), causing a mismatch.
    sample_gap = N  # was: N * 5

    for step in range(sample_steps):
        dH, accepted = mc_step(theta, T, L, Psi, Delta)

        if step % sample_gap == 0:
            mx, my = compute_mxy(theta)
            m = math.sqrt(mx*mx + my*my)

            mx_acc += mx
            my_acc += my
            m_acc  += m

            # FIX 2: EPR = log(w_reverse / w_forward), summed over accepted moves only.
            # Old code used dH/T, which is NOT the entropy production rate.
            # Correct formula: Bandini Eq.(21), Shi Eq.(6).
            #   w_f = 0.5*(1 - tanh( dH/(2T)))
            #   w_r = 0.5*(1 - tanh(-dH/(2T))) = 0.5*(1 + tanh(dH/(2T)))
            # Note: we accumulate only on accepted steps where dH != 0 is meaningful.
            if accepted:
                w_f = 0.5 * (1.0 - math.tanh( dH / (2.0*T)))
                w_r = 0.5 * (1.0 - math.tanh(-dH / (2.0*T)))
                if w_f > 0:
                    epr_acc += math.log(w_r / w_f)

            count += 1

    mx_avg  = mx_acc  / count
    my_avg  = my_acc  / count
    m_avg   = m_acc   / count
    epr_avg = epr_acc / count

    return {
        "m":   m_avg,
        "mx":  mx_avg,
        "my":  my_avg,
        "EPR": epr_avg
    }