"""Train the value-of-computation predictor.

We predict the *gain* a gap gets from extra compute, not its absolute error.

That choice is forced by the structure of the problem. For a fixed gap the
error at the cheapest budget, E0, does not depend on the budget we pick, so

    minimise  sum_g E[g, b(g)]   ==   maximise  sum_g ( E[g,0] - E[g,b(g)] )

and the E0 terms cancel. Predicting absolute error per budget and letting the
knapsack difference two noisy predictions is strictly worse: the differences
are dominated by prediction noise and the allocator then exploits that noise,
which is the optimizer's curse. Measured directly -- a per-budget absolute
error model loses to the uniform knob by 35-50%, while predicting gain wins.

Predicted gains are made monotone in cost (more compute cannot help less),
which removes the remaining way the allocator could chase noise.

Reads results/delta_dataset.npz, writes results/router.pkl.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.oracle import oracle_frontier, allocate_lagrangian, uniform_choice  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/delta_dataset.npz")
    ap.add_argument("--out", default="results/router.pkl")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=27)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, E, budgets = d["X"], d["errors"], d["budgets"]
    names = [str(n) for n in d["feature_names"]]
    costs = (budgets[:, 0] * budgets[:, 1]).astype(float)
    n, B = E.shape
    print(f"gaps={n}  features={X.shape[1]}  budgets={list(costs.astype(int))}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = int(n * args.test_frac)
    te, tr = perm[:n_test], perm[n_test:]
    print(f"train={len(tr)}  held-out={len(te)}")

    # --- is there anything to learn? oracle bound first -------------------
    print("\n=== ORACLE BOUND on held-out gaps (go/no-go) ===")
    front = oracle_frontier(E[te], costs)
    print(f"{'nfe/gap':>9} {'uniform MAE':>12} {'oracle MAE':>11} {'gain %':>8} {'escal.':>7}")
    for r in front:
        print(f"{r['budget_per_gap']:9.1f} {r['uniform_MAE']:12.3f} "
              f"{r['oracle_MAE']:11.3f} {r['oracle_gain_pct']:7.1f}% "
              f"{r['escalation_rate']*100:6.0f}%")
    best_gain = max(r["oracle_gain_pct"] for r in front)
    print(f"\nbest oracle gain over the uniform knob: {best_gain:.1f}%")
    if best_gain < 2.0:
        print("VERDICT: per-gap allocation cannot beat the global knob even with "
              "perfect information. Report as a negative result.")
    else:
        print("VERDICT: headroom exists; a learned router is worth building.")

    # --- learned router ---------------------------------------------------
    from lightgbm import LGBMRegressor
    # Target: gain over the cheapest budget. Budget 0 has gain 0 by definition,
    # so only budgets 1..B-1 need a model.
    G = E[:, 0:1] - E                      # (n, B), G[:,0] == 0
    models, r2s = [None], [float("nan")]
    print("\n=== gain models (target: error reduction vs cheapest budget) ===")
    for b in range(1, B):
        m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=args.seed, verbose=-1)
        m.fit(X[tr], G[tr, b])
        pred = m.predict(X[te])
        ss_res = float(((G[te, b] - pred) ** 2).sum())
        ss_tot = float(((G[te, b] - G[te, b].mean()) ** 2).sum())
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        models.append(m)
        r2s.append(r2)
        print(f"  budget {int(budgets[b,0]):2d}x{int(budgets[b,1]):2d} "
              f"({int(costs[b]):4d} nfe)  held-out R2 on gain = {r2:+.3f}")

    def predict_gain(Xm):
        g = np.zeros((Xm.shape[0], B))
        for b in range(1, B):
            g[:, b] = models[b].predict(Xm)
        g = np.maximum(g, 0.0)                  # extra compute cannot hurt
        return np.maximum.accumulate(g, axis=1)  # monotone in cost

    # The allocator minimises, so hand it negative predicted gain.
    pred_E = -predict_gain(X[te])

    print("\n=== LEARNED ROUTER vs UNIFORM vs ORACLE (held-out) ===")
    rows = []
    for r in front:
        budget = r["budget_per_gap"] * len(te)
        idx, cost, _ = allocate_lagrangian(pred_E, costs, budget)
        router_mae = float(E[te][np.arange(len(te)), idx].mean())
        rows.append({
            "budget_per_gap": r["budget_per_gap"],
            "uniform_MAE": r["uniform_MAE"],
            "router_MAE": router_mae,
            "oracle_MAE": r["oracle_MAE"],
            "router_gain_pct": 100.0 * (r["uniform_MAE"] - router_mae) / max(r["uniform_MAE"], 1e-12),
            "oracle_gain_pct": r["oracle_gain_pct"],
            "router_cost": cost,
            "escalation_rate": float(np.mean(idx > 0)),
        })
    print(f"{'nfe/gap':>9} {'uniform':>9} {'router':>9} {'oracle':>9} "
          f"{'router gain':>12} {'of oracle':>10}")
    for r in rows:
        capture = (100.0 * r["router_gain_pct"] / r["oracle_gain_pct"]
                   if r["oracle_gain_pct"] > 0.01 else float("nan"))
        print(f"{r['budget_per_gap']:9.1f} {r['uniform_MAE']:9.3f} "
              f"{r['router_MAE']:9.3f} {r['oracle_MAE']:9.3f} "
              f"{r['router_gain_pct']:11.1f}% {capture:9.0f}%")

    imp = np.mean([m.feature_importances_ for m in models if m is not None], axis=0)
    order = np.argsort(-imp)
    print("\n=== top features by mean importance ===")
    for i in order[:8]:
        print(f"  {names[i]:26s} {imp[i]:7.1f}")

    Path("results").mkdir(exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"models": models, "costs": costs, "budgets": budgets,
                     "feature_names": names, "r2": r2s, "target": "gain"}, f)
    Path("results/router_eval.json").write_text(json.dumps(
        {"oracle_frontier": front, "router_rows": rows, "r2": r2s,
         "feature_importance": {names[i]: float(imp[i]) for i in order}}, indent=2))
    print(f"\nwrote {args.out} and results/router_eval.json")


if __name__ == "__main__":
    main()
