# -*- coding: utf-8 -*-
"""
mxlpy port of the NPQ / light-quality photosynthesis model.

DESIGN NOTES
------------
mxlpy builds models as a directed acyclic graph of named `derived` quantities and `reactions`,
each a plain top-level Python function referencing other named quantities
by string (`args=[...]`). This works cleanly for everything in the model
*except* the electron-transport/carbon-assimilation block, which has a
genuine circular dependency (eta depends on Cc, Cc depends on A, A depends
on J_P680_forward, which depends on D_P680_abs, which depends on eta) that
was solved in the ODE version via fixed-point iteration -- that iteration
can't be expressed as a directed acyclic graph of `add_derived` calls, since there's no
"solve this system" primitive in mxlpy's derived-quantity mechanism.

The fix: keep the whole QSS iteration as ONE cached function (`_qss_block`,
module-level + `functools.lru_cache`), and expose each of its outputs to
mxlpy as a separate thin top-level "extractor" function. mxlpy will call
several of these extractors with the *same* arguments during a single
right-hand-side evaluation (once per derived quantity that needs them);
the cache means only the first call in that batch actually does the work,
the rest are O(1) hits. Cache eviction (`maxsize=32`) keeps memory bounded
over a long integration, since we only need very recent evaluations
cached, not the whole trajectory.

Every extractor function must be a real top-level `def` (not a factory-
generated closure or lambda) so that mxlpy can pickle it, per mxlpy's own
requirement ("functions need to be pickle-able, lambdas not supported").
"""

from __future__ import annotations

from functools import lru_cache, partial

import numpy as np
from mxlpy import Model, Simulator, fns, make_protocol
from mxlpy.integrators import Scipy

# ---------------------------------------------------------------------------
# Spectra: kept as plain module-level globals (like in your original script)
# rather than mxlpy parameters, since they're fixed 371-wavelength arrays,
# not scalars to be fit/scanned. Populate these via `set_spectra(...)`
# before building/simulating the model.
# ---------------------------------------------------------------------------
""" This synthax is used to tell that each parameter is either type numpy array or
None and is subsequently initiated as None. It does not enforce anything but
it is for readability"""
spectrum_0: np.ndarray | None = None
spectrum_1: np.ndarray | None = None
spectrum_LHCII: np.ndarray | None = None
spectrum_PSI_LHCI: np.ndarray | None = None
spectrum_CSM: np.ndarray | None = None
leaf_absorbance: np.ndarray | None = None
PAM_spect: np.ndarray | None = None
absorbance_photopigments: np.ndarray | None = None


def set_spectra(
    *,
    spectrum_0_: np.ndarray,
    spectrum_1_: np.ndarray,
    spectrum_LHCII_: np.ndarray,
    spectrum_PSI_LHCI_: np.ndarray,
    spectrum_CSM_: np.ndarray,
    leaf_absorbance_: np.ndarray,
    PAM_spect_: np.ndarray,
    absorbance_photopigments_: np.ndarray,
) -> None:
    """Call this once at startup with your real loaded spectra (from
    load_spectrum / the parameters.xlsx / Hogewoning CSVs), before building
    or simulating the model.
    This function sets the different spectra as global parameters so they can be used everywhere."""
    global spectrum_0, spectrum_1, spectrum_LHCII, spectrum_PSI_LHCI
    global spectrum_CSM, leaf_absorbance, PAM_spect, absorbance_photopigments
    spectrum_0 = spectrum_0_
    spectrum_1 = spectrum_1_
    spectrum_LHCII = spectrum_LHCII_
    spectrum_PSI_LHCI = spectrum_PSI_LHCI_
    spectrum_CSM = spectrum_CSM_
    leaf_absorbance = leaf_absorbance_
    PAM_spect = PAM_spect_
    absorbance_photopigments = absorbance_photopigments_
    _qss_block.cache_clear()  # stale cache would reference old spectra


# Specifies all the parameters used in the QSS calculation. The order needs to be
#maintained throughout
QSS_ARGS = [
    "light_id", "LHCII_per_PSII", "LHCII_per_PSI",
    "Ci", "D_P680_tot", "D_P700_tot", "PQ_pool", "K_N2",
    "K_P2", "K_D2", "K_F2", "K_U2", "K_P1", "K_F1",
    "rate_tot_P700_open", "rate_tot_P700_closed",
    "gamma_star", "eta_CET", "eta_LET", "CB6F_recovery_rate",
    "Vcmax", "MM_constant_carboxy", "chloroplastic_oxygen", "MM_constant_oxy",
    "Rd", "g_meso",
    "epsilon_F1", "epsilon_F2",
]

"""lru_cache reduces the amount of calculations that is needed for running the model.
It saves the output of the function add a given input parameter set. Repeated calls 
with the same input gets the output from memory, instead of recalculating. 
(https://www.geeksforgeeks.org/python/python-functools-lru_cache/)"""

@lru_cache(maxsize=32)
def _qss_block(
    light_id, LHCII_per_PSII, LHCII_per_PSI, 
    Ci, D_P680_tot, D_P700_tot, PQ_pool, K_N2,
    K_P2, K_D2, K_F2, K_U2, K_P1, K_F1,
    rate_tot_P700_open, rate_tot_P700_closed,
    gamma_star, eta_CET, eta_LET, CB6F_recovery_rate,
    Vcmax, MM_constant_carboxy, chloroplastic_oxygen, MM_constant_oxy,
    Rd, g_meso,
    epsilon_F1, epsilon_F2,
) -> tuple[float, ...]:
    """Solves the fast electron-transport/carbon-assimilation subsystem
    to self-consistency for one instantaneous state + light spectrum.
    
    n_iter and tol define how many iterative steps are taken and how small a step 
    needs to be before a solution is accepted. 
    
    relax makes it that the iterative steps are set slightly smaller. This reduces
    problems after sudden big shifts in the steady state and prevents overshoots 
    into unrealistic values which can give mathematical errors."""
    n_iter, tol, relax = 200, 1e-10, 0.3

    spectrum = spectrum_0 if light_id < 0.5 else spectrum_1

    
    

    rate_tot_P680_open = K_P2 + K_N2 + K_D2 + K_F2 + K_U2
    rate_tot_P680_closed = K_N2 + K_D2 + K_F2 + K_U2
    open_sharing_rate = K_U2 / rate_tot_P680_open
    closed_sharing_rate = K_U2 / rate_tot_P680_closed
    Phi_P680_photo = K_P2 / rate_tot_P680_open
    Phi_P700_photo = K_P1 / rate_tot_P700_open

    PSI_LHCII_spectrum_model = D_P700_tot * LHCII_per_PSI * spectrum_LHCII
    PSI_LHCI_absorbance_spectrum = D_P700_tot * spectrum_PSI_LHCI
    absorptance_PSI = PSI_LHCII_spectrum_model + PSI_LHCI_absorbance_spectrum

    PSII_core_spectrum_model = D_P680_tot * spectrum_CSM
    PSII_LHCII_spectrum_model = D_P680_tot * (LHCII_per_PSII - 2) * spectrum_LHCII
    absorptance_PSII = PSII_core_spectrum_model + PSII_LHCII_spectrum_model

    sigma_II = absorptance_PSII / (absorptance_PSI + absorptance_PSII)
    sigma_I = absorptance_PSI / (absorptance_PSI + absorptance_PSII)
    P680_excitation = float(np.sum(spectrum * leaf_absorbance * sigma_II))
    P700_excitation = float(np.sum(spectrum * leaf_absorbance * sigma_I))

    Cc = Ci * 0.7
    D_P680_abs = 0.5 * D_P680_tot
    J_P680_forward = 0.0
    J_P700_forward = 0.0
    A = 0.0
    PQH2_abs = 0.0
    D_P700_abs = 0.0
    converged = False

    for _ in range(n_iter):
        phi_U2 = ((D_P680_abs / D_P680_tot) * open_sharing_rate
                  + (1 - (D_P680_abs / D_P680_tot)) * closed_sharing_rate)

        a = 3 + 7 * (gamma_star / Cc)
        b = 4 + 8 * (gamma_star / Cc) * eta_CET
        eta = 1 - (eta_LET / eta_CET) + (a / b)



        a_val = PQ_pool * Phi_P680_photo * P680_excitation * eta
        b_val = (CB6F_recovery_rate * (1 - phi_U2)) + (Phi_P680_photo * P680_excitation * eta)
        PQH2_abs = a_val / b_val
        PQH2_abs = min(max(PQH2_abs, 0.0), PQ_pool)

        D_P700_abs = (PQH2_abs * CB6F_recovery_rate * D_P700_tot) / (PQ_pool * Phi_P700_photo * P700_excitation)
        D_P680_abs_target = (1 - (PQH2_abs / PQ_pool)) * D_P680_tot

        if D_P700_abs > D_P700_tot:
            D_P680_abs_target = (((Phi_P700_photo * P700_excitation) / eta)
                                  * ((1 - phi_U2) / (Phi_P680_photo * P680_excitation)) * D_P680_tot)
            PQH2_abs = (1 - (D_P680_abs_target / D_P680_tot)) * PQ_pool
        D_P680_abs_target = min(max(D_P680_abs_target, 0.0), D_P680_tot)
        D_P700_abs = min(D_P700_tot, D_P700_abs)

        J_P680_forward = D_P680_abs_target * Phi_P680_photo * P680_excitation / (D_P680_tot * (1 - phi_U2))
        J_P700_forward = D_P700_abs * Phi_P700_photo * P700_excitation / D_P700_tot

        W_J = (Cc * J_P680_forward) / (4 * Cc + 8 * gamma_star)
        W_c = (Cc * Vcmax) / (Cc + MM_constant_carboxy * (1 + (chloroplastic_oxygen / MM_constant_oxy)))
        A_new = (1 - (gamma_star / Cc)) * min(W_c, W_J) - Rd
        Cc_target = Ci - (A_new / g_meso)
        Cc_target = min(max(Cc_target, 1e-3), Ci - 1e-6)

        D_P680_abs_new = D_P680_abs + relax * (D_P680_abs_target - D_P680_abs)
        Cc_new = Cc + relax * (Cc_target - Cc)

        if abs(Cc_new - Cc) < tol and abs(D_P680_abs_new - D_P680_abs) < tol * D_P680_tot:
            Cc, D_P680_abs, A = Cc_new, D_P680_abs_new, A_new
            converged = True
            break
        Cc, D_P680_abs, A = Cc_new, D_P680_abs_new, A_new

    if not converged:
        raise RuntimeError(
            "QSS fixed-point iteration did not converge "
            f"(Cc={Cc}, D_P680_abs={D_P680_abs}, PQH2_abs={PQH2_abs})."
        )

    D_P680 = D_P680_abs / D_P680_tot
    D_P700 = D_P700_abs / D_P700_tot
    PQH2 = PQH2_abs / PQ_pool

    excitation_PSII_PAM = float(np.sum(PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_II))
    excitation_PSI_PAM = float(np.sum(PAM_spect * leaf_absorbance * absorbance_photopigments * sigma_I))
    Fm = (excitation_PSII_PAM * epsilon_F2 * (K_F2 / (rate_tot_P680_closed - K_U2))
          + excitation_PSI_PAM * epsilon_F1 * (K_F1 / rate_tot_P700_closed))

    return (A, Cc, D_P680, D_P700, PQH2, Fm, D_P680_abs,
            P680_excitation, P700_excitation, J_P680_forward, J_P700_forward)


""" Extractors -- one real top-level function per output, all sharing the
 same cached computation. Order must match the tuple returned above.
 This extracts all the relevant outputs of the algebraic block"""
 
 #arguments are directly passed on to the _qss_block call
def get_A(*args: float) -> float:
    return _qss_block(*args)[0]


def get_Cc(*args: float) -> float:
    return _qss_block(*args)[1]


def get_D_P680(*args: float) -> float:
    return _qss_block(*args)[2]


def get_D_P700(*args: float) -> float:
    return _qss_block(*args)[3]


def get_PQH2(*args: float) -> float:
    return _qss_block(*args)[4]


def get_Fm(*args: float) -> float:
    return _qss_block(*args)[5]


def get_D_P680_abs(*args: float) -> float:
    return _qss_block(*args)[6]


def get_P680_excitation(*args: float) -> float:
    return _qss_block(*args)[7]


def get_P700_excitation(*args: float) -> float:
    return _qss_block(*args)[8]


def get_J_P680_forward(*args: float) -> float:
    return _qss_block(*args)[9]


def get_J_P700_forward(*args: float) -> float:
    return _qss_block(*args)[10]





# ---------------------------------------------------------------------------
# Everything downstream of the QSS block is a genuine directed acyclic graph -- plain
# add_derived / add_reaction calls, no iteration needed.
# ---------------------------------------------------------------------------
def lhcii_flux(LHCII_per_PSII, PQH2, D_P680_tot, PQ_pool,
               dephosphorylation_rate_constant, phosphorylation_rate_constant,
               LHCII_per_PSI, D_P700_tot):
    LHCII_dephosphorylated = LHCII_per_PSII * D_P680_tot
    LHCII_phosphorylated = LHCII_per_PSI * D_P700_tot
    dephosphorylation_rate = LHCII_phosphorylated * dephosphorylation_rate_constant
    phosphorylation_rate = LHCII_dephosphorylated * phosphorylation_rate_constant * PQH2
    return phosphorylation_rate - dephosphorylation_rate

def lhcii_to_psi_rate(lhcii_net_flux, D_P700_tot):
    return lhcii_net_flux / D_P700_tot


def lhcii_to_psii_rate(lhcii_net_flux, D_P680_tot):
    return -lhcii_net_flux / D_P680_tot


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------
def build_model(params: dict) -> Model:
    """`params` is the flat params dict, minus the spectra arrays. These are set
    using `set_spectra(...)` separately ."""
    scalar_params = {k: v for k, v in params.items() if np.isscalar(v)}
    scalar_params.setdefault("light_id", 0.0)
    
    #Creates the model with the given parameters and initial states for the 7 state variables
    m = (
        Model()
        .add_parameters(scalar_params)
        .add_variables({
            "LHCII_per_PSII": 3.0, #reasonable value from literature see Jans2026
            "LHCII_per_PSI": 1.0, #reasonable value from literature see Jans2026
        })
    )

    # QSS-block extractors
    #loops over calculated outputs and adds those as derived variables. 
    for name, fn in [
        ("A", get_A), ("Cc", get_Cc), ("D_P680", get_D_P680), ("D_P700", get_D_P700),
        ("PQH2", get_PQH2), ("Fm", get_Fm), ("D_P680_abs", get_D_P680_abs),
        ("P680_excitation", get_P680_excitation), ("P700_excitation", get_P700_excitation),
        ("J_P680_forward", get_J_P680_forward), ("J_P700_forward", get_J_P700_forward),
    ]:
        m = m.add_derived(name, fn, args=QSS_ARGS)

    m = (
        m.add_derived("lhcii_net_flux", lhcii_flux, args=[
            "LHCII_per_PSII", "PQH2", "D_P680_tot", "PQ_pool",
            "dephosphorylation_rate_constant", "phosphorylation_rate_constant",
            "LHCII_per_PSI", "D_P700_tot",
        ])
        .add_derived("v_LHCII_to_PSI", lhcii_to_psi_rate, args=["lhcii_net_flux", "D_P700_tot"])
        .add_derived("v_LHCII_to_PSII", lhcii_to_psii_rate, args=["lhcii_net_flux", "D_P680_tot"])
    )

    m = (
        m.add_reaction("v_LHCII_PSI", fns.constant, args=["v_LHCII_to_PSI"],
                        stoichiometry={"LHCII_per_PSI": 1})
        .add_reaction("v_LHCII_PSII", fns.constant, args=["v_LHCII_to_PSII"],
                       stoichiometry={"LHCII_per_PSII": 1})
    )
    return m


def build_protocol(section_minutes: list, section_light_ids: list) -> "pd.DataFrame":  # noqa: F821
    """Each step's duration is in SECONDS -- mxlpy protocols assign 1 second of
    Timedelta per 1 time unit of integration, which lines up exactly with
    your s^-1 rate constants (no Time_min/Time_sec conversion needed)."""
    light_protocol = [(section_minutes[i] * 60, {"light_id": section_light_ids[i]}) for i in range(len(section_minutes))]
    return make_protocol(light_protocol)

"""TThis makes the simulator for the model. It uses the Scipy integrator as this
works well with the algebraic block that is needed. rtol and atol are the relative
and absolute tolerance for the error

Partial is used so that the solves can be set without actually running the Simulator yet. 
"""
def make_simulator(model: Model, atol: float = 1e-9, rtol: float = 1e-6,
                    method: str = "LSODA") -> Simulator:
    return Simulator(model, integrator=partial(Scipy, atol=atol, rtol=rtol, method=method))
