import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from thermodynamics import (
    N_eq,
    rho_eq,
    drho_eq_dT,
    Gamma_htophiphi,
    Gamma_htophiphi_energy
)


def Y_obs_DM(ms):
    rho_DM_today = 9.711808033846503e-48  # GeV^4
    T0_GeV = 2.72548 * 8.617333262145e-14
    s0 = (2.0 * 43.0 * np.pi**2 / (11.0 * 45.0)) * T0_GeV**3
    return rho_DM_today / (ms * s0)


def Y_solver_from_Ns(cosmo, Ns, afinal):
    T_final = float(cosmo.T_of_a(afinal))
    s_final = float(cosmo.entropy_density(T_final))
    N_final = float(Ns(afinal))
    return N_final / (afinal**3 * s_final)


def abundance_ratio(cosmo, Ns, afinal, ms):
    Y_sol = Y_solver_from_Ns(cosmo, Ns, afinal)
    Y_obs = Y_obs_DM(ms)
    return Y_sol / Y_obs


def solve_condensate_N_loga_quad(H_of_a, N_init, ai, af, n_eval=4000, zero_rel=1e-14):
    u_grid = np.linspace(np.log(ai), np.log(af), n_eval)
    a_grid = np.exp(u_grid)

    integrand = np.array([1.0 / float(H_of_a(a)) for a in a_grid])
    du = u_grid[1] - u_grid[0]

    I = np.zeros_like(u_grid)
    I[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * du)

    def build(Gamma_X):
        N_grid = N_init * np.exp(-Gamma_X * I)
        threshold = zero_rel * N_init
        N_grid = np.where(N_grid < threshold, 0.0, N_grid)
        N_interp = PchipInterpolator(u_grid, N_grid, extrapolate=True)

        def N_of_a(a):
            val = np.asarray(N_interp(np.log(a)))
            return np.where(val < threshold, 0.0, val)

        def n_of_a(a):
            a = np.asarray(a, dtype=float)
            val = np.asarray(N_interp(np.log(a))) / a**3
            return np.where(val < threshold / np.maximum(a**3, 1e-300), 0.0, val)

        return N_grid, N_of_a, n_of_a

    return a_grid, u_grid, build


def two_body_energy_at_rest(m_parent, m_daughter, m_other=0.0):
    return (m_parent**2 + m_daughter**2 - m_other**2) / (2.0 * m_parent)


def condensate_number_source(nX, Gamma_X, multiplicity=1.0):
    return multiplicity * Gamma_X * nX


def condensate_energy_source(nX, Gamma_X, E0, multiplicity=1.0):
    return multiplicity * Gamma_X * nX * E0


def solve_free_in_loga(
    cosmo,
    ms,
    lhs,
    xi_inf,
    ai,
    af,
    mh=125.0,
    v=246.0,
    nX_of_a=None,
    Gamma_X=None,
    mX=None,
    m_other=0.0,
    multiplicity_X=1.0,
    rtol=1e-6,
    atol=1e-9,
    method="BDF",
    max_step_u=0.01,
    a_match=None,   # NEW: hand-chosen matching scale for NR tail
):
    """
    Solve the cBE in u = log(a), optionally only up to a_match and then use
    a piecewise asymptotic tail:
        N(a) = const
        T(a) = T_match * (a_match / a)^2
    for a > a_match.
    """
    if a_match is not None:
        if a_match <= ai:
            raise ValueError("a_match must satisfy a_match > ai.")
        a_stop = min(af, a_match)
    else:
        a_stop = af

    ui = np.log(ai)
    uf = np.log(a_stop)

    def rhs_u(u, y):
        a = np.exp(u)
        N, Ts = y

        Ts = max(Ts, 1e-300)

        Tsm = float(cosmo.T_of_a(a))
        Hsm = float(cosmo.H_of_a(a))

        # ---------- source 1: thermal parent ----------
        src1_N = (lhs**2) * Gamma_htophiphi(Tsm, ms, mh=mh, v=v)
        src1_E = (lhs**2) * Gamma_htophiphi_energy(Tsm, ms, mh=mh, v=v)

        # ---------- source 2: condensate at rest ----------
        src2_N = 0.0
        src2_E = 0.0
        if nX_of_a is not None and Gamma_X is not None and mX is not None:
            nX = float(nX_of_a(a))
            E0 = two_body_energy_at_rest(mX, ms, m_other=m_other)

            src2_N = condensate_number_source(
                nX=nX,
                Gamma_X=Gamma_X,
                multiplicity=multiplicity_X,
            )
            src2_E = condensate_energy_source(
                nX=nX,
                Gamma_X=Gamma_X,
                E0=E0,
                multiplicity=multiplicity_X,
            )

        dN_da = (a**2 / Hsm) * (src1_N + src2_N)

        Neq = max(N_eq(a, Ts, ms), 1e-300)
        rho = rho_eq(Ts, ms)
        drho = drho_eq_dT(Ts, ms)

        numerator = (
            (src1_E + src2_E) / (a * Hsm)
            - (3.0 / a) * (N * Ts / a**3)
            - (dN_da / Neq) * rho
        )

        denominator = (
            (N / Neq) * drho
            - a**3 * (rho / (Ts * Neq))**2 * N
        )

        dTs_da = numerator / denominator

        dN_du = a * dN_da
        dTs_du = a * dTs_da

        return [dN_du, dTs_du]

    Tini = float(cosmo.T_of_a(ai))
    Ts_ini = xi_inf * Tini
    N_ini = N_eq(ai, Ts_ini, ms)

    sol = solve_ivp(
        rhs_u,
        (ui, uf),
        [N_ini, Ts_ini],
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=max_step_u,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    # attach tail metadata for piecewise continuation
    sol._piecewise_tail = None
    if a_match is not None and af > a_match:
        N_match, T_match = sol.sol(np.log(a_match))
        sol._piecewise_tail = {
            "enabled": True,
            "a_match": float(a_match),
            "N_match": float(N_match),
            "T_match": float(T_match),
            "af_requested": float(af),
        }

    return sol


def make_a_functions_from_loga_solution(sol):
    a_ref = float(np.exp(sol.t[-1]))
    N_ref = max(float(sol.y[0, -1]), 1e-300)
    T_ref = max(float(sol.y[1, -1]), 1e-300)

    def Ns(a):
        a_in = np.asarray(a, dtype=float)
        a_arr = np.atleast_1d(a_in)

        out = np.empty_like(a_arr, dtype=float)

        mask_low = a_arr <= a_ref
        mask_high = ~mask_low

        if np.any(mask_low):
            vals = np.asarray(sol.sol(np.log(a_arr[mask_low]))[0], dtype=float)
            out[mask_low] = np.maximum(vals, 1e-300)
        if np.any(mask_high):
            out[mask_high] = N_ref

        return out.item() if np.ndim(a_in) == 0 else out

    def Ts(a):
        a_in = np.asarray(a, dtype=float)
        a_arr = np.atleast_1d(a_in)

        out = np.empty_like(a_arr, dtype=float)

        mask_low = a_arr <= a_ref
        mask_high = ~mask_low

        if np.any(mask_low):
            vals = np.asarray(sol.sol(np.log(a_arr[mask_low]))[1], dtype=float)
            out[mask_low] = np.maximum(vals, 1e-300)
        if np.any(mask_high):
            out[mask_high] = T_ref * (a_ref / a_arr[mask_high])**2

        return out.item() if np.ndim(a_in) == 0 else out

    return Ns, Ts


def solve_free_in_loga_with_abundance(
    cosmo,
    ms,
    lhs,
    xi_inf,
    ai,
    af,
    mh=125.0,
    v=246.0,
    nX_of_a=None,
    Gamma_X=None,
    mX=None,
    m_other=0.0,
    multiplicity_X=1.0,
    rtol=1e-6,
    atol=1e-9,
    method="BDF",
    max_step_u=0.01,
    a_match=None,
):
    sol = solve_free_in_loga(
        cosmo=cosmo,
        ms=ms,
        lhs=lhs,
        xi_inf=xi_inf,
        ai=ai,
        af=af,
        mh=mh,
        v=v,
        nX_of_a=nX_of_a,
        Gamma_X=Gamma_X,
        mX=mX,
        m_other=m_other,
        multiplicity_X=multiplicity_X,
        rtol=rtol,
        atol=atol,
        method=method,
        max_step_u=max_step_u,
        a_match=a_match,
    )

    Ns, Ts = make_a_functions_from_loga_solution(sol)
    Y_sol = Y_solver_from_Ns(cosmo, Ns, af)
    Y_obs = Y_obs_DM(ms)
    ratio = Y_sol / Y_obs

    return sol, Ns, Ts, Y_sol, Y_obs, ratio


def rho_X_of_a(a, mX, nX_of_a):
    return mX * float(nX_of_a(a))


def radiation_dominance_ratio(a, cosmo, mX, nX_of_a):
    rhoX = rho_X_of_a(a, mX, nX_of_a)
    rhor = float(cosmo.rho_rad(float(cosmo.T_of_a(a))))
    return rhoX / rhor
