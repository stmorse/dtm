"""
WS1 / R1.1 step 7: multiple testing (R2.6) and step diagnostics (R2.5).

Takes the locked drift-test results and

  R2.6  applies Benjamini-Hochberg FDR across all tested groups, for Lambda
        and for delta^2, and reports counts at q < 0.05 and q < 0.01.

  R2.5  reports diagnostics on the step vectors. The permutation test assumes
        neither Gaussianity nor a random walk (its null is exchangeability in
        time), so these are descriptive:
        - lag-1 autocorrelation rho1 = sum <d_t, d_t+1> / sum ||d_t||^2.
          Independent noise around a fixed centre makes the steps MA(1) with
          rho1 -> -0.5; a random walk with independent increments gives 0.
          Each group's rho1 is also z-scored against its own shuffle null,
          since a finite exchangeable sequence has E[rho1] < 0.
        - normality of the projected steps (Shapiro-Wilk per dimension).

Run from src/:
    python 8f_diagnostics.py --seed 0
"""

import argparse
import configparser
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import drift

BLUE, ORANGE, GRAY, INK, INK2, SURFACE = '#2a78d6', '#eb6834', '#c9c8c3', '#0b0b0b', '#52514e', '#fcfcfb'


def to_md(df):
    """Minimal markdown table (avoids the optional tabulate dependency)."""
    cols = list(df.columns)
    rows = ['| ' + ' | '.join(map(str, cols)) + ' |',
            '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        rows.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(rows)


def bh_qvalues(p):
    """Benjamini-Hochberg q-values for a 1-d array of p-values (NaNs kept)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    x = p[ok]
    m = len(x)
    order = np.argsort(x)
    ranked = x[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def rho1(steps):
    """Lag-1 autocorrelation of a sequence of step vectors."""
    num = float(np.sum(steps[:-1] * steps[1:]))
    den = float(np.sum(steps * steps))
    return num / den if den > 0 else np.nan


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--results', type=str,
                        default='../output/ws1/8b/asis_group_half_r5_s0_B100k/groups.csv')
    parser.add_argument('--run_subpath', type=str, default='mbkm_50')
    parser.add_argument('--align_subpath', type=str, default='align/max_100')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--r', type=int, default=5)
    parser.add_argument('--n_perm_rho', type=int, default=1000)
    parser.add_argument('--out_path', type=str, default='../output/ws1/8f_diagnostics')
    parser.add_argument('--tab_path', type=str, default='../refs/revision/tables')
    parser.add_argument('--fig_path', type=str,
                        default='../refs/revision/figs/R2.5_step_diagnostics_draft')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.tab_path, exist_ok=True)
    run_path = os.path.join(g['save_path'], args.run_subpath)
    align_path = os.path.join(run_path, args.align_subpath)

    df = pd.read_csv(args.results).set_index('coi')

    # ---- R2.6: BH-FDR ----------------------------------------------------
    for col in ['p_lam', 'p_d2']:
        if col in df:
            df['q' + col[1:]] = bh_qvalues(df[col].values)

    counts = []
    for col, qcol, name in [('p_lam', 'q_lam', 'Lambda'), ('p_d2', 'q_d2', 'delta^2')]:
        if col not in df:
            continue
        counts.append({'statistic': name, 'n_tested': int(df[col].notna().sum()),
                       'p<0.05': int((df[col] < 0.05).sum()),
                       'p<0.01': int((df[col] < 0.01).sum()),
                       'q<0.05': int((df[qcol] < 0.05).sum()),
                       'q<0.01': int((df[qcol] < 0.01).sum())})
    counts = pd.DataFrame(counts)

    # ---- R2.5: step diagnostics -----------------------------------------
    C = drift.load_centroids(run_path, k=args.k)
    labels = drift.load_group_labels(align_path)
    rng = np.random.default_rng(args.seed)

    rows, shapiro_all = [], []
    for coi in df.index:
        rows_k, _ = drift.group_rows(labels, coi, k=args.k)
        X = drift.trajectory(C, rows_k)
        T = X.shape[0]
        r_k = max(1, min(args.r, (T - 2) // 2))
        Xr = drift.group_basis(X, r_k)
        steps_full, steps_r = np.diff(X, axis=0), np.diff(Xr, axis=0)

        obs = rho1(steps_full)
        null = np.array([rho1(np.diff(X[rng.permutation(T)], axis=0))
                         for _ in range(args.n_perm_rho)])
        z = (obs - null.mean()) / null.std(ddof=1) if null.std(ddof=1) > 0 else np.nan

        # Shapiro-Wilk on each projected step dimension (needs >= 3 points)
        sh = [stats.shapiro(steps_r[:, j]).pvalue for j in range(r_k)] if len(steps_r) >= 3 else []
        shapiro_all.extend(sh)
        rows.append({'coi': coi, 'T': T, 'r': r_k,
                     'rho1': obs, 'rho1_null_mean': float(null.mean()), 'rho1_z': z,
                     'shapiro_min': min(sh) if sh else np.nan,
                     'shapiro_frac_lt05': float(np.mean(np.array(sh) < 0.05)) if sh else np.nan})
    diag = pd.DataFrame(rows).set_index('coi')
    out = df.join(diag[['rho1', 'rho1_null_mean', 'rho1_z', 'shapiro_min', 'shapiro_frac_lt05']])
    out.to_csv(os.path.join(args.out_path, 'groups.csv'))

    summary = {
        'n_groups': len(out),
        'rho1_median': float(out.rho1.median()),
        'rho1_iqr': [float(out.rho1.quantile(.25)), float(out.rho1.quantile(.75))],
        'rho1_null_median': float(out.rho1_null_mean.median()),
        'rho1_z_median': float(out.rho1_z.median()),
        'frac_rho1_below_-0.3': float((out.rho1 < -0.3).mean()),
        'shapiro_frac_dims_lt05': float(out.shapiro_frac_lt05.mean()),
        'fdr': counts.to_dict('records'),
    }
    with open(os.path.join(args.out_path, 'params.json'), 'w') as f:
        json.dump(vars(args) | {'summary': summary}, f, indent=2)

    counts.to_csv(os.path.join(args.tab_path, 'R2.6_fdr_counts_draft.csv'), index=False)
    with open(os.path.join(args.tab_path, 'R2.6_fdr_counts_draft.md'), 'w') as f:
        f.write('# R2.6 - multiple testing (BH-FDR), locked R1.1 procedure\n\n')
        f.write(to_md(counts) + '\n')

    np.save(os.path.join(args.out_path, 'shapiro_all.npy'), np.array(shapiro_all))
    plot_diagnostics(out, np.array(shapiro_all), args)
    print(counts.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'fdr'}, indent=2))


def plot_diagnostics(out, shapiro_all, args):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in axs:
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRAY)
        ax.tick_params(colors=INK2, labelsize=8)

    ax = axs[0]
    ax.hist(out.rho1.dropna(), bins=30, color=BLUE, alpha=0.85)
    ax.axvline(-0.5, color=ORANGE, lw=2, ls='--', label='−0.5: independent noise (MA(1))')
    ax.axvline(0.0, color=INK2, lw=2, ls=':', label='0: independent RW increments')
    ax.axvline(out.rho1.median(), color=INK, lw=2, label=f'median {out.rho1.median():.2f}')
    ax.set_xlabel('lag-1 autocorrelation of steps', color=INK2)
    ax.set_ylabel('topic groups', color=INK2)
    ax.set_title('(a) Step autocorrelation', fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=8)

    ax = axs[1]
    n_bins = 20
    ax.hist(shapiro_all, bins=np.linspace(0, 1, n_bins + 1), color=BLUE, alpha=0.85)
    ax.axhline(len(shapiro_all) / n_bins, color=ORANGE, lw=2, ls='--',
               label='uniform (exact normality)')
    frac = float(np.mean(shapiro_all < 0.05))
    ax.set_xlabel(f'Shapiro–Wilk p per step dimension ({frac:.1%} below 0.05)', color=INK2)
    ax.set_ylabel('group × dimension', color=INK2)
    ax.set_title('(b) Normality of projected steps', fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle('DRAFT: step diagnostics for the aligned topic groups (R2.5)',
                 fontsize=10, color=INK, x=0.005, ha='left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(f'{args.fig_path}.{ext}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
