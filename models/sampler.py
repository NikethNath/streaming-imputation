"""Budget-controllable sampling for a trained CSDI model.

PyPOTS samples CSDI with the full ancestral DDPM chain and loops over posterior
samples one at a time. Both are fixed costs, and both are exactly what the
router needs to vary. This module reimplements the reverse process over the
*already trained* weights so that:

  * the number of reverse steps is free (DDIM-style strided sampling over the
    model's own cumulative-alpha schedule), and
  * posterior samples are drawn in one batched pass instead of a serial loop.

Cost is reported as network function evaluations, NFE = steps x samples, which
is the hardware-independent unit the whole project optimises. CSDI's published
defaults are 50 x 100 = 5,000 NFE per window.

`sample(..., mode="ancestral", n_steps=T)` reproduces the PyPOTS reverse loop
exactly and is used to verify that the strided sampler is a faithful
generalisation rather than a different model.
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------
# input assembly -- mirrors pypots.imputation.csdi TestDatasetForCSDI +
# _assemble_input_for_testing, so the sampler sees exactly what predict() sees.
# --------------------------------------------------------------------------
def prepare_inputs(X: np.ndarray, device="cpu"):
    """X: (B, L, K) with NaN for missing -> observed (B,K,L), cond_mask, tp."""
    observed = ~np.isnan(X)
    filled = np.nan_to_num(X, nan=0.0).astype(np.float32)
    B, L, K = X.shape
    obs = torch.from_numpy(filled).permute(0, 2, 1).to(device)
    cond = torch.from_numpy(observed.astype(np.float32)).permute(0, 2, 1).to(device)
    tp = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(0).expand(B, L)
    return obs, cond, tp


def step_schedule(total_steps: int, n_steps: int) -> np.ndarray:
    """Evenly spaced subsequence of diffusion indices, ascending, ending at T-1."""
    n_steps = int(max(1, min(n_steps, total_steps)))
    if n_steps == total_steps:
        return np.arange(total_steps)
    return np.unique(np.linspace(0, total_steps - 1, n_steps).round().astype(int))


@torch.no_grad()
def sample(csdi, X: np.ndarray, n_steps: int | None = None, n_samples: int = 8,
           mode: str = "ddim", eta: float = 0.0, device: str = "cpu",
           batch_samples: bool = True, crn_seed: int | None = None):
    """Draw posterior samples for a batch of windows under a chosen budget.

    `crn_seed` fixes the initial latent x_T (common random numbers). With
    eta=0 the DDIM trajectory is deterministic given x_T, so a fixed seed makes
    the whole sampler a deterministic function of the input. That matters
    because the per-gap *gain* from extra compute is otherwise dominated by the
    randomness of x_T rather than by any property of the gap -- see
    scripts/gain_reliability.py. Under common random numbers the same latent is
    reused across budgets, so differences between budgets reflect the budget
    alone.

    Returns (samples, nfe) where samples has shape
    (n_samples, B, L, K) in the same units as X, with observed cells preserved.
    """
    core = csdi.model.model          # _CSDI
    backbone = core.backbone
    core.eval()

    total = backbone.n_diffusion_steps
    taus = np.arange(total) if mode == "ancestral" else step_schedule(total, n_steps or total)
    alpha = np.asarray(backbone.alpha, dtype=np.float64)        # cumulative alpha-bar
    alpha_hat = np.asarray(backbone.alpha_hat, dtype=np.float64)  # per-step 1-beta
    beta = np.asarray(backbone.beta, dtype=np.float64)

    obs, cond, tp = prepare_inputs(X, device)
    B, K, L = obs.shape
    side = core.get_side_info(tp, cond)

    reps = n_samples if batch_samples else 1
    n_rounds = 1 if batch_samples else n_samples

    def expand(t, r):
        return t.repeat_interleave(r, dim=0) if r > 1 else t

    obs_r, cond_r, side_r = expand(obs, reps), expand(cond, reps), expand(side, reps)

    out = []
    for _round in range(n_rounds):
        if crn_seed is not None:
            g = torch.Generator(device=device).manual_seed(int(crn_seed) + _round)
            x = torch.randn(B * reps, K, L, device=device, generator=g)
        else:
            x = torch.randn(B * reps, K, L, device=device)
        for i in range(len(taus) - 1, -1, -1):
            t = int(taus[i])
            cond_obs = (cond_r * obs_r).unsqueeze(1)
            noisy_target = ((1.0 - cond_r) * x).unsqueeze(1)
            diff_input = torch.cat([cond_obs, noisy_target], dim=1)
            eps = backbone.diff_model(diff_input, side_r,
                                      torch.tensor([t], device=device))

            if mode == "ancestral":
                c1 = 1 / alpha_hat[t] ** 0.5
                c2 = (1 - alpha_hat[t]) / (1 - alpha[t]) ** 0.5
                x = c1 * (x - c2 * eps)
                if t > 0:
                    sigma = ((1.0 - alpha[t - 1]) / (1.0 - alpha[t]) * beta[t]) ** 0.5
                    x = x + sigma * torch.randn_like(x)
            else:
                a_t = alpha[t]
                a_prev = alpha[int(taus[i - 1])] if i > 0 else 1.0
                x0 = (x - (1 - a_t) ** 0.5 * eps) / a_t ** 0.5
                sigma = (eta * ((1 - a_prev) / (1 - a_t)) ** 0.5
                         * (1 - a_t / a_prev) ** 0.5) if (eta > 0 and i > 0) else 0.0
                dir_xt = max(1 - a_prev - sigma ** 2, 0.0) ** 0.5 * eps
                x = a_prev ** 0.5 * x0 + dir_xt
                if sigma > 0:
                    x = x + sigma * torch.randn_like(x)

        imputed = obs_r + x * (1.0 - cond_r)          # keep observed cells
        out.append(imputed.view(B, reps, K, L))

    samples = (out[0] if batch_samples else torch.cat(out, dim=1))
    samples = samples.permute(1, 0, 3, 2).cpu().numpy()   # (n_samples, B, L, K)
    nfe = int(len(taus) * n_samples)
    return samples, nfe


def point_estimate(samples: np.ndarray) -> np.ndarray:
    """Posterior median over the sample axis."""
    return np.median(samples, axis=0)


def nfe(n_steps: int, n_samples: int) -> int:
    return int(n_steps) * int(n_samples)
