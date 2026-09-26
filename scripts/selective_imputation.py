"""Selective imputation: know which gaps not to trust.

Our central finding is an asymmetry. The *gain* a gap gets from extra compute is
not predictable from context (held-out R2 < 0, and it stays negative even under
common random numbers). But a gap's *error* is: it is reproducible across
independent sampling runs (r = 0.76..0.91) and gradient-boosted models reach
positive held-out R2 on it.

That asymmetry kills adaptive compute but enables something else. If we can
predict which imputations will be bad, the system can decline to publish them.
For an air-quality alerting service that is the more valuable capability: a
missing reading is honest, whereas a confidently wrong reading can place a
station in the wrong public-health band.

This script measures the risk-coverage trade-off. Gaps are ranked by predicted
error; we withhold the worst and report the error among those we keep.
Comparators: random abstention (no skill) and oracle abstention (perfect skill).

Writes results/selective_imputation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/delta_dataset.npz")
    ap.add_argument("--budget-idx", type=int, default=4,
                    help="which budget column to study (default 20x8)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=27)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, E, budgets = d["X"], d["errors"], d["budgets"]
    names = [str(n) for n in d["feature_names"]]
    b = args.budget_idx
    S, N = int(budgets[b][0]), int(budgets[b][1])
    y = E[:, b]
    n = len(y)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_te = int(n * args.test_frac)
    te, tr = perm[:n_te], perm[n_te:]
    print(f"budget {S}x{N} ({S*N} NFE)   gaps: train={len(tr)} held-out={len(te)}")

    from lightgbm import LGBMRegressor
    m = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                      min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                      random_state=args.seed, verbose=-1)
    m.fit(X[tr], y[tr])
    pred = m.predict(X[te])
    true = y[te]

    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    spearman = float(np.corrcoef(np.argsort(np.argsort(pred)),
                                np.argsort(np.argsort(true)))[0, 1])
    print(f"error model: held-out R2 = {r2:+.3f}, rank corr = {spearman:+.3f}")

    order_model = np.argsort(pred)            # keep lowest predicted error
    order_oracle = np.argsort(true)
    base = float(true.mean())

    rows = []
    print(f"\n{'coverage':>9} {'model':>8} {'oracle':>8} {'random':>8} "
          f"{'vs base':>9}")
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(te))))
        mae_model = float(true[order_model[:k]].mean())
        mae_oracle = float(true[order_oracle[:k]].mean())
        # random abstention: expectation is the base rate
        mae_random = base
        rows.append({"coverage": cov, "model_MAE": mae_model,
                     "oracle_MAE": mae_oracle, "random_MAE": mae_random,
                     "reduction_pct": 100 * (base - mae_model) / base})
        print(f"{cov*100:8.0f}% {mae_model:8.2f} {mae_oracle:8.2f} "
              f"{mae_random:8.2f} {rows[-1]['reduction_pct']:8.1f}%")

    # How much of the achievable selectivity did we capture at 80% coverage?
    r80 = [r for r in rows if abs(r["coverage"] - 0.8) < 1e-9][0]
    capture = (100 * (base - r80["model_MAE"]) /
               max(base - r80["oracle_MAE"], 1e-12))
    print(f"\nat 80% coverage the model captures {capture:.0f}% of the "
          f"oracle's achievable error reduction")

    imp = m.feature_importances_
    top = np.argsort(-imp)[:6]
    print("\ntop features for predicting error:")
    for i in top:
        print(f"  {names[i]:26s} {imp[i]:6.0f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/selective_imputation.json").write_text(json.dumps(
        {"budget": f"{S}x{N}", "nfe": S * N, "r2": r2, "rank_corr": spearman,
         "base_MAE": base, "rows": rows, "capture_at_80pct": capture,
         "top_features": {names[i]: float(imp[i]) for i in top}}, indent=2))
    print("\nwrote results/selective_imputation.json")


if __name__ == "__main__":
    main()
