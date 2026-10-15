"""CSDI baseline via PyPOTS.

We use PyPOTS rather than the 2021 reference repository because it is
maintained, pip-installable and exposes CSDI, SAITS and BRITS behind one API.
The course brief permits "at least a simplified version" of the base method, so
exact reproduction of the published CRPS is not required -- but the defaults
below are the ones from CSDI's own config/base.yaml (50 diffusion steps, quad
schedule, beta 1e-4 .. 0.5) so we stay close.

The two knobs the Phase 2 router allocates are both reachable from here:
  * n_diffusion_steps -- chain length, fixed at construction
  * n_sampling_times  -- posterior samples, free at inference

INTERFACE FROZEN 21 Sep. Tracks C and D build against impute(); do not change
its signature without telling them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CSDI_DEFAULTS = dict(
    n_layers=4,
    n_heads=8,
    n_channels=64,
    d_time_embedding=128,
    d_feature_embedding=16,
    d_diffusion_embedding=128,
    n_diffusion_steps=50,
    schedule="quad",
    beta_start=0.0001,
    beta_end=0.5,
    target_strategy="random",
)


class CSDIBaseline:
    """Thin wrapper giving a stable train/impute surface over PyPOTS CSDI."""

    def __init__(self, n_steps: int, n_features: int, device: str = "cpu",
                 epochs: int = 100, batch_size: int = 16, patience: int | None = 10,
                 saving_path: str = "checkpoints", lr: float = 1e-3,
                 lr_milestones: list[int] | None = None, **overrides):
        from pypots.imputation import CSDI
        from pypots.optim import Adam

        params = dict(CSDI_DEFAULTS)
        params.update(overrides)
        # CSDI's own recipe decays the learning rate 10x late in training. The
        # PyPOTS scheduler steps once per optimiser step, so milestones are in
        # batches, not epochs.
        scheduler = None
        if lr_milestones:
            from pypots.optim.lr_scheduler import MultiStepLR
            scheduler = MultiStepLR(milestones=lr_milestones, gamma=0.1)
        self.n_steps = n_steps
        self.n_features = n_features
        self.params = params
        self.model = CSDI(
            n_steps=n_steps,
            n_features=n_features,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device=device,
            saving_path=saving_path,
            model_saving_strategy="better",   # save on every improvement, so a crash keeps progress
            optimizer=Adam(lr=lr, lr_scheduler=scheduler),
            **params,
        )

    # -- training ---------------------------------------------------------
    def train(self, train_windows: np.ndarray, val_windows: np.ndarray | None = None,
              val_mask_rate: float = 0.1, seed: int = 27):
        """train_windows: (n_windows, n_steps, n_features), NaN where missing.

        The validation set needs X to be a *corrupted* copy and X_ori the intact
        one -- PyPOTS scores only the cells that differ. Passing the same array
        as both gives a validation loss of exactly 0.0 every epoch, which makes
        early stopping fire immediately and keeps epoch 1 as "best".
        """
        train_set = {"X": train_windows}
        val_set = None
        if val_windows is not None:
            rng = np.random.default_rng(seed)
            observed = ~np.isnan(val_windows)
            hide = (rng.random(val_windows.shape) < val_mask_rate) & observed
            val_corrupt = val_windows.copy()
            val_corrupt[hide] = np.nan
            val_set = {"X": val_corrupt, "X_ori": val_windows}
        self.model.fit(train_set, val_set)
        return self

    # -- inference --------------------------------------------------------
    def impute(self, windows: np.ndarray, n_samples: int = 100):
        """Impute a batch of windows.

        Returns (point_estimate, samples) where point_estimate is the posterior
        median with shape (n_windows, n_steps, n_features) and samples has shape
        (n_samples, n_windows, n_steps, n_features).
        """
        out = self.model.predict({"X": windows}, n_sampling_times=n_samples)
        samples = out["imputation"]           # (n_windows, n_samples, steps, feats)
        samples = np.asarray(samples)
        if samples.ndim == 4:
            samples = np.transpose(samples, (1, 0, 2, 3))
        else:                                  # n_samples == 1
            samples = samples[None, ...]
        point = np.median(samples, axis=0)
        # Observed entries are never touched by the model; clamp them anyway so
        # downstream code can treat the output as a complete series.
        observed = ~np.isnan(windows)
        point = np.where(observed, np.nan_to_num(windows), point)
        return point, samples

    # -- persistence ------------------------------------------------------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path, overwrite=True)

    def load(self, path: str):
        self.model.load(path)
        return self


def nfe_per_window(n_diffusion_steps: int, n_samples: int) -> int:
    """Network function evaluations for one window.

    This is the hardware-independent cost unit the whole project optimises.
    CSDI's published defaults are 50 x 100 = 5,000 evaluations per window.
    """
    return int(n_diffusion_steps) * int(n_samples)
