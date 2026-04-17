import numpy as np
from numba import njit
import math

# =========================
# NEIGHBOR STRUCTURE
# =========================
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

    theta_i   : spin angle
    direction : bond direction psi_ij
    Psi       : half-angle of vision cone for this spin
    """
    return abs(angle_diff(direction, theta_i)) <= Psi


# =========================
# LOCAL ΔH  (two-cone version)
# =========================
# Each site (x,y) has its own half-cone angle Psi_arr[x,y].
# J_ij(theta_i) = 1 iff in_cone(theta_i, psi_ij, Psi_arr[x,y]).
# J_ji(theta_j) = 1 iff in_cone(theta_j, psi_ji, Psi_arr[nx,ny]).
#
# The Hamiltonian structure is identical to the single-cone case:
#   H_SS = -sum_{<ij>} [J_ij + J_ji] cos(theta_i - theta_j)
#   H_Sa = -sum_{<ij>} [J_ij cos(theta_j - phi_i) + J_ji cos(theta_i - phi_j)]
# with phi_i = theta_i - pi (mirror constraint).

@njit
def delta_H(theta, Psi_arr, x, y, theta_new, L):
    """
    Compute ΔH = H(theta_new, phi_old) - H(theta_old, phi_old).
    Psi_arr[i,j] holds the per-site half-cone angle.
    """
    old_theta = theta[x, y]
    phi_old   = old_theta - np.pi   # auxiliary spin, held fixed during proposal

    Psi_i = Psi_arr[x, y]          # half-cone of site being updated

    dH = 0.0

    for k in range(4):
        nx = (x + dx[k]) % L
        ny = (y + dy[k]) % L

        theta_j = theta[nx, ny]
        phi_j   = theta_j - np.pi

        Psi_j = Psi_arr[nx, ny]    # half-cone of neighbor

        direction_ij = psi_dir[k]
        direction_ji = psi_dir[(k + 2) % 4]

        Jij_old = 1.0 if in_cone(old_theta, direction_ij, Psi_i) else 0.0
        Jij_new = 1.0 if in_cone(theta_new, direction_ij, Psi_i) else 0.0
        Jji     = 1.0 if in_cone(theta_j,   direction_ji, Psi_j) else 0.0  # unchanged

        # H_SS
        old_ss = -(Jij_old + Jji) * math.cos(old_theta - theta_j)
        new_ss = -(Jij_new + Jji) * math.cos(theta_new - theta_j)

        # H_Sa
        old_sa = -Jij_old * math.cos(theta_j - phi_old) \
                 -Jji     * math.cos(old_theta - phi_j)
        new_sa = -Jij_new * math.cos(theta_j - phi_old) \
                 -Jji     * math.cos(theta_new - phi_j)

        dH += (new_ss + new_sa) - (old_ss + old_sa)

    return dH


# =========================
# MC STEP
# =========================

@njit
def mc_step(theta, Psi_arr, T, L, Delta):
    """
    Individual-update constrained Glauber step.
    Psi_arr: (L,L) float64 array of per-site half-cone angles.
    """
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)

    delta     = (np.random.rand() * 2 - 1) * Delta
    theta_new = theta[x, y] + delta

    if theta_new > 2*np.pi:
        theta_new -= 2*np.pi
    elif theta_new < 0:
        theta_new += 2*np.pi

    dH   = delta_H(theta, Psi_arr, x, y, theta_new, L)
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
    L  = theta.shape[0]
    for i in range(L):
        for j in range(L):
            mx += math.cos(theta[i, j])
            my += math.sin(theta[i, j])
    mx /= (L*L)
    my /= (L*L)
    return mx, my


# =========================
# CONE-MASK BUILDERS  (pure Python, called before JIT loop)
# =========================

def make_cone_mask_random(L, Psi1, Psi2, seed=None):
    """
    Each site independently assigned Psi1 or Psi2 with probability 0.5 each.

    Parameters
    ----------
    L    : lattice side length
    Psi1 : half-cone angle for type-1 spins (radians)
    Psi2 : half-cone angle for type-2 spins (radians)
    seed : optional RNG seed for reproducibility

    Returns
    -------
    Psi_arr : (L,L) float64 array
    mask    : (L,L) bool array  (True = type-1, False = type-2)
    """
    rng  = np.random.default_rng(seed)
    mask = rng.integers(0, 2, size=(L, L)).astype(bool)   # True/False 50-50
    Psi_arr = np.where(mask, Psi1, Psi2).astype(np.float64)
    return Psi_arr, mask


def make_cone_mask_rows(L, Psi1, Psi2):
    """
    Row-block assignment (5 rows thick):
      rows 0–4   → Psi1
      rows 5–9   → Psi2
      rows 10–14 → Psi1
      ...
    """
    rows = np.arange(L)

    # divide rows into blocks of 5, then alternate blocks
    mask = ((rows // 5) % 2 == 0)[:, np.newaxis] * np.ones((1, L), dtype=bool)

    Psi_arr = np.where(mask, Psi1, Psi2).astype(np.float64)
    return Psi_arr, mask


# =========================
# INNER SIMULATION LOOP  (Numba JIT)
# =========================

@njit
def _run_loop(theta, Psi_arr, T, L, Delta, burn_steps, sample_steps, sample_gap):
    mx_acc  = 0.0
    my_acc  = 0.0
    m_acc   = 0.0
    epr_acc = 0.0
    count   = 0

    for _ in range(burn_steps):
        mc_step(theta, Psi_arr, T, L, Delta)

    for step in range(sample_steps):
        dH, accepted = mc_step(theta, Psi_arr, T, L, Delta)

        if step % sample_gap == 0:
            mx, my = compute_mxy(theta)
            m = math.sqrt(mx*mx + my*my)

            mx_acc += mx
            my_acc += my
            m_acc  += m

            if accepted:
                w_f = 0.5 * (1.0 - math.tanh( dH / (2.0*T)))
                w_r = 0.5 * (1.0 - math.tanh(-dH / (2.0*T)))
                if w_f > 0:
                    epr_acc += math.log(w_r / w_f)

            count += 1

    return mx_acc/count, my_acc/count, m_acc/count, epr_acc/count


# =========================
# MAIN API
# =========================

def run_simulation(L, T, theta_deg, init="disordered",
                   theta_deg2=None, cone_assign="single",
                   seed=None):
    """
    Parameters
    ----------
    L          : lattice side length
    T          : temperature
    theta_deg  : full vision-cone angle for type-1 spins (degrees).
                 If cone_assign="single", ALL spins use this cone.
    theta_deg2 : full vision-cone angle for type-2 spins (degrees).
                 Required when cone_assign != "single".
    cone_assign: "single"   – uniform cone (original behaviour)
                 "random"   – each spin independently 50/50 assigned type-1 or type-2
                 "rows"     – even rows → type-1, odd rows → type-2
    seed       : RNG seed for random assignment (ignored otherwise)

    Returns
    -------
    dict with keys: m, mx, my, EPR, Psi_arr, mask
    """
    Psi1  = np.deg2rad(theta_deg) / 2.0
    Delta = 0.5

    # ---- build per-site cone array ----
    if cone_assign == "single":
        Psi_arr = np.full((L, L), Psi1, dtype=np.float64)
        mask    = np.ones((L, L), dtype=bool)

    elif cone_assign == "random":
        if theta_deg2 is None:
            raise ValueError("theta_deg2 must be provided for cone_assign='random'")
        Psi2    = np.deg2rad(theta_deg2) / 2.0
        Psi_arr, mask = make_cone_mask_random(L, Psi1, Psi2, seed=seed)

    elif cone_assign == "rows":
        if theta_deg2 is None:
            raise ValueError("theta_deg2 must be provided for cone_assign='rows'")
        Psi2    = np.deg2rad(theta_deg2) / 2.0
        Psi_arr, mask = make_cone_mask_rows(L, Psi1, Psi2)

    else:
        raise ValueError(f"Unknown cone_assign='{cone_assign}'. "
                         "Choose 'single', 'random', or 'rows'.")

    # ---- initialise spins ----
    if init == "disordered":
        theta = np.random.rand(L, L) * 2 * np.pi
    else:
        theta = np.zeros((L, L))

    # ---- schedule ----
    total_steps  = int(20 * L**4)
    burn_steps   = int(0.3 * total_steps)
    sample_steps = int(0.7 * total_steps)
    sample_gap   = L * L

    mx_avg, my_avg, m_avg, epr_avg = _run_loop(
        theta, Psi_arr, T, L, Delta, burn_steps, sample_steps, sample_gap
    )

    return {
        "m":       m_avg,
        "mx":      mx_avg,
        "my":      my_avg,
        "EPR":     epr_avg,
        "Psi_arr": Psi_arr,   # (L,L) per-site half-cone angles
        "mask":    mask        # (L,L) bool: True = type-1
    }