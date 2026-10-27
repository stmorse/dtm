"""
Self-test for src/matching.py. No I/O, no data, runs in under a second:

    python 10z_matching_check.py

Checks the identities and the three structural cases WS2 asked for, plus the
two traps this module exists to prevent (noise labels wrapping in
matched_purity, and homogeneity read without its null).
"""

import numpy as np

import matching as M


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not cond:
        raise AssertionError(name)


def main():
    rng = np.random.default_rng(0)

    print('pair_scores(a, a) is a perfect score')
    a = rng.integers(0, 40, size=4000)
    s = M.pair_scores(a, a)
    for key in ['purity_1to1', 'overlap_many_to_one_median', 'ari', 'ami',
                'homogeneity', 'completeness']:
        check(f'{key} == 1', abs(s[key] - 1.0) < 1e-12, f'{s[key]:.6f}')

    print('a random relabelling scores at chance')
    b = rng.permutation(a)
    s = M.pair_scores(a, b)
    check('|ari| < 0.01', abs(s['ari']) < 0.01, f"{s['ari']:+.5f}")
    check('|ami| < 0.01', abs(s['ami']) < 0.01, f"{s['ami']:+.5f}")
    check('homogeneity is NOT ~0 (it is not chance-adjusted)',
          s['homogeneity'] > 0.05, f"{s['homogeneity']:.4f}")
    s2 = M.pair_scores(a, b, rng=np.random.default_rng(1))
    check('null keys appear when rng is passed',
          'homogeneity_null' in s2 and
          abs(s2['homogeneity'] - s2['homogeneity_null']) < 0.05,
          f"obs {s2['homogeneity']:.4f} null {s2['homogeneity_null']:.4f}")

    print('matched_purity agrees with pair_scores on noise-free k-means labels')
    x = rng.integers(0, 10, size=2000)
    y = x.copy()
    flip = rng.random(2000) < 0.2        # one mask, used once
    y[flip] = rng.integers(0, 10, size=int(flip.sum()))
    p, _ = M.matched_purity(x, y, 10)
    check('purity matches pair_scores purity_1to1',
          abs(p - M.pair_scores(x, y)['purity_1to1']) < 1e-12, f'{p:.4f}')

    print('matched_purity refuses noise labels instead of wrapping them')
    try:
        M.matched_purity(np.array([0, 1, -1]), np.array([0, 1, 2]), 3)
        check('raises on -1', False)
    except ValueError as e:
        check('raises on -1', 'outside' in str(e))

    print('match_many_to_one: one merge, one split, one group sent to noise')
    #   a: 0,0,0,0 | 1,1,1,1 | 2,2,2,2 | 3,3,3,3
    #   b: group 0 and 1 both -> 10 (merge); 2 splits 3/1; 3 all noise
    A = np.array([0]*4 + [1]*4 + [2]*4 + [3]*4)
    B = np.array([10]*4 + [10]*4 + [20]*3 + [21] + [-1]*4)
    mt = M.match_many_to_one(A, B).set_index('g_a')
    check('0 and 1 both match b group 10 (merge)',
          mt.loc[0, 'g_b'] == 10 and mt.loc[1, 'g_b'] == 10)
    check('merge overlap is 1.0 both ways',
          mt.loc[0, 'overlap'] == 1.0 and mt.loc[1, 'overlap'] == 1.0)
    check('2 splits: overlap 0.75', abs(mt.loc[2, 'overlap'] - 0.75) < 1e-12,
          f"{mt.loc[2, 'overlap']:.4f}")
    check('3 is entirely noise: frac_to_noise 1.0, g_b -1, overlap NaN',
          mt.loc[3, 'frac_to_noise'] == 1.0 and mt.loc[3, 'g_b'] == -1
          and np.isnan(mt.loc[3, 'overlap']))
    check('n_a counts all members, n_a_core only non-noise',
          mt.loc[3, 'n_a'] == 4 and mt.loc[3, 'n_a_core'] == 0)
    check('many-to-one hides the merge (median overlap 1.0 for a shattered b)',
          M.match_many_to_one(A, B).overlap.median() == 1.0)

    print('match_many_to_one: weights change the overlap as intended')
    Aw = np.array([0, 0, 0, 0])
    Bw = np.array([10, 10, 20, 20])
    w = np.array([1.0, 1.0, 1.0, 97.0])          # the 4th member dominates
    check('unweighted overlap 0.5',
          abs(M.match_many_to_one(Aw, Bw).overlap.iloc[0] - 0.5) < 1e-12)
    mw = M.match_many_to_one(Aw, Bw, weights=w)
    check('weighted overlap 0.98 and g_b flips to 20',
          abs(mw.overlap.iloc[0] - 0.98) < 1e-12 and mw.g_b.iloc[0] == 20,
          f'{mw.overlap.iloc[0]:.4f}')

    print('noise_summary')
    ns = M.noise_summary(np.array([-1, -1, 0, 1]), np.array([-1, 0, 0, -1]))
    check('fractions', ns['frac_noise_a'] == 0.5 and ns['frac_noise_b'] == 0.5)
    check('jaccard_noise 1/3', abs(ns['jaccard_noise'] - 1/3) < 1e-12)

    print('containment: min vs directional, and the Jaccard trap')
    sa, sb = set(range(18)), set(range(115))      # perfectly nested, R2.1 sizes
    check('mode=min is 1.0 under nesting', M.containment(sa, sb) == 1.0)
    check('mode=a is 1.0, mode=a reversed is 18/115',
          M.containment(sa, sb, 'a') == 1.0
          and abs(M.containment(sb, sa, 'a') - 18/115) < 1e-12)
    jac = len(sa & sb) / len(sa | sb)
    check('Jaccard caps at 0.157 on the same nested sets -- why we avoid it',
          abs(jac - 18/115) < 1e-12, f'{jac:.4f}')

    print('enrichment_test')
    et = M.enrichment_test(24, 77, 0.151)
    check('enrichment ~2.06', abs(et['enrichment'] - (24/77)/0.151) < 1e-12,
          f"{et['enrichment']:.3f}")
    check('p is small', et['p'] < 1e-3, f"p={et['p']:.2e}")
    check('no enrichment at the base rate',
          M.enrichment_test(12, 100, 0.12)['p'] > 0.4)

    print('\nAll matching.py self-tests passed.')


if __name__ == '__main__':
    main()
