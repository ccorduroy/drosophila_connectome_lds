"""
Figures for the Dec 1 talk (spec section 15: F2 simulation checks, F3 hidden-region r against
fraction hidden, F4 per-fly paired differences), drawn only from committed results:
  results/simulations_v2.json   G2 under protocol v2 (S1-S3)
  results/r3_tierA_v3.json      tier A-prime under protocol v3 (grid-edge extension)

    python make_talk_figures.py            # -> figures/talk_F2_sim_checks.png, talk_F3_hidden_r.png, talk_F4_paired.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TEXT, TEXT2, GRID = '#191919', '#666666', '#e6e6e6'
BLUE, ORANGE, AQUA = '#257de0', '#f47230', '#00b894'       # categorical slots 1-3 (validated, all pairs)
GRAY, GRAY_L = '#8c8c8c', '#bdbdbd'
SESOI = 0.02
FR = ('0.25', '0.5', '0.75')
plt.rcParams.update({'font.size': 11, 'text.color': TEXT, 'axes.labelcolor': TEXT2, 'xtick.color': TEXT2,
                     'ytick.color': TEXT2, 'axes.edgecolor': GRID, 'axes.titleweight': 'bold',
                     'axes.titlesize': 12, 'axes.titlelocation': 'left', 'savefig.dpi': 200,
                     'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})


def _clean(ax, xgrid=False):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis='x' if xgrid else 'y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def f2(sim, out):
    S1 = sim['S1']['R3']
    panels = [
        ('S1: planted connectome effect', [(f"M3 − M4, {int(float(f) * 100)}%", S1[f]['M3-M4'], ORANGE) for f in FR]
         + [(f"M3 − M2, {int(float(f) * 100)}%", S1[f]['M3-M2'], AQUA) for f in FR], 'detects it'),
        ('S2: no connectome effect', [("M3 − M4, 50%", sim['S2']['0.5']['M3-M4'], ORANGE),
                                      ("M3 − M2, 50%", sim['S2']['0.5']['M3-M2'], AQUA)], 'invents none'),
        ('S3: permuted connectome', [("matching perm. − M3, 50%", sim['S3']['0.5']['M4match-M3'], GRAY),
                                     ("M3 − M2, 50%", sim['S3']['0.5']['M3-M2'], AQUA)], 'real wiring not favoured'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True,
                             gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.95})
    for ax, (title, rows, verdict) in zip(axes, panels):
        y = np.arange(len(rows))[::-1]
        for yi, (lab, s, c) in zip(y, rows):
            ax.plot(s['ci'], [yi, yi], color=c, lw=2.5, solid_capstyle='round')
            ax.plot(s['hl'], yi, 'o', ms=8, color=c, mec='white', mew=1.5, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=TEXT)
        ax.axvline(0, color=TEXT2, lw=1)
        ax.axvline(SESOI, color=TEXT2, lw=1, ls=(0, (4, 3)))
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(title, pad=24)
        ax.text(0, 1.015, verdict, transform=ax.transAxes, color=TEXT2, fontsize=10, va='bottom')
        _clean(ax, xgrid=True)
        ax.set_xlim(-0.012, 0.042)
        ax.set_xlabel('paired difference in hidden-region r')
    for ax in axes:
        ax.text(SESOI, -0.62, ' 0.02', color=TEXT2, fontsize=9, va='bottom')
    fig.text(0.0, -0.08, "Hodges–Lehmann median over 20 simulated flies with 95% bootstrap interval (most intervals are "
             "narrower than the dots). Dashed line: 0.02, the smallest effect of interest. Protocol v2 "
             "(results/simulations_v2.json).", color=TEXT2, fontsize=9)
    fig.savefig(out); plt.close(fig)


def f3(v3, out):
    x = [25, 50, 75]
    med = lambda m: [np.median(list(v3['score_smooth'][f][m].values())) for f in FR]
    base = lambda b: [np.median(list(v3['baselines_r'][f][b].values())) for f in FR]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'width_ratios': [1, 1.35], 'wspace': 0.45})
    # left: full range, H1
    for m, c, lw in (('M3', BLUE, 2.5), ('M1', GRAY, 2), ('M0', GRAY_L, 2)):
        a1.plot(x, med(m), '-o', color=c, lw=lw, ms=8, mec='white', mew=1.5)
    a1.text(76.5, med('M3')[-1], 'M3 connectome', color=TEXT, va='center', fontsize=10)
    a1.text(76.5, 0.035, 'M0, M1 (no coupling\nto hidden regions)', color=TEXT2, va='bottom', fontsize=10)
    a1.set_ylim(-0.08, 0.72); a1.set_xlim(20, 100)
    a1.set_title('Hidden regions become reconstructable (H1)')
    # right: zoom, H2 and baselines
    for m, c, lab in (('M3', BLUE, 'M3 connectome'), ('M2', AQUA, 'M2 uniform'), ('M4', ORANGE, 'M4 shuffled (mean of 5)')):
        a2.plot(x, med(m), '-o', color=c, lw=2.5 if m == 'M3' else 2, ms=8, mec='white', mew=1.5, zorder=3)
    for b, ls, lab in (('B6', (0, (5, 3)), 'B6 wiring-weighted average'), ('B8', (0, (1, 2)), 'B8 uniform average')):
        a2.plot(x, base(b), color=TEXT2, lw=1.5, ls=ls)
    labs = [(med('M3')[-1], 'M3 connectome', TEXT), (base('B6')[-1], 'B6 wiring average', TEXT2),
            (med('M2')[-1], 'M2 uniform', TEXT), (med('M4')[-1], 'M4 shuffled (mean of 5)', TEXT),
            (base('B8')[-1], 'B8 uniform average', TEXT2)]
    labs.sort(key=lambda t: -t[0]); ys = []
    for yv, lab, c in labs:                      # nudge apart so labels never collide
        yv2 = yv if not ys or ys[-1] - yv >= 0.0075 else ys[-1] - 0.0075
        ys.append(yv2); a2.text(76.5, yv2, lab, color=c, va='center', fontsize=10)
    a2.set_ylim(0.49, 0.635); a2.set_xlim(20, 108)
    a2.set_title('The real wiring beats generic coupling (H2), zoomed')
    for ax in (a1, a2):
        ax.set_xticks(x); ax.set_xticklabels([f'{v}%' for v in x], color=TEXT)
        ax.set_xlabel('regions hidden in every fly'); ax.set_ylabel('hidden-region r (median over 20 flies)')
        _clean(ax)
    fig.text(0.0, -0.06, "Smoothed estimates, Pearson r per hidden region, Fisher-z over regions and 5 hidden sets; "
             "tier A′, protocol v3 (results/r3_tierA_v3.json). Note the different y-scales.", color=TEXT2, fontsize=9)
    fig.savefig(out); plt.close(fig)


def f4(v3, out):
    order = sorted(v3['score_smooth']['0.25']['M3'])
    sc = v3['score_smooth']
    tests = {r['test']: r for r in v3['tests'] if 'parts' in r}
    fig, ax = plt.subplots(figsize=(12, 4.4))
    rng = np.random.default_rng(675)
    ticks, ticklabels = [], []
    for gi, f in enumerate(FR):
        t = tests[f"H2 {int(float(f) * 100)}%"]
        for ci, (b, c) in enumerate((('M4', ORANGE), ('M2', AQUA))):
            xc = gi * 3 + ci
            d = np.array([sc[f]['M3'][fl] - sc[f][b][fl] for fl in order])
            ax.scatter(xc + rng.uniform(-0.18, 0.18, len(d)), d, s=22, color=c, alpha=0.55, lw=0)
            s = t['parts'][f'M3-{b}']
            ax.plot([xc + 0.32] * 2, s['ci'], color=TEXT, lw=2, solid_capstyle='round')
            ax.plot(xc + 0.32, s['hl'], 'o', ms=8, color=c, mec=TEXT, mew=1.2, zorder=3)
            ticks.append(xc); ticklabels.append(f"M3 − {b}")
        verdict = 'supported' if t['supported'] else 'not supported:\nmedian below 0.02'
        ax.text(gi * 3 + 0.5, 0.062, f"{int(float(f) * 100)}% hidden", ha='center', fontweight='bold', fontsize=11)
        ax.text(gi * 3 + 0.5, 0.057, verdict, ha='center', va='top', color=TEXT2, fontsize=10)
    ax.axhline(0, color=TEXT2, lw=1)
    ax.axhline(SESOI, color=TEXT2, lw=1, ls=(0, (4, 3)))
    ax.text(8.0, SESOI, ' smallest effect\n of interest (0.02)', color=TEXT2, fontsize=9, va='center')
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, color=TEXT)
    ax.set_xlim(-0.6, 9.2); ax.set_ylim(-0.012, 0.068)
    ax.set_ylabel('per-fly difference in hidden-region r')
    _clean(ax)
    fig.text(0.0, -0.06, "Dots: the 20 flies. Marker and bar: Hodges–Lehmann median with 95% bootstrap interval. "
             "Holm p = 1.3e-5 for every H2 test; supported also needs median ≥ 0.02. Tier A′, protocol v3.",
             color=TEXT2, fontsize=9)
    fig.savefig(out); plt.close(fig)


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    sim = json.load(open('results/simulations_v2.json'))['results']
    v3 = json.load(open('results/r3_tierA_v3.json'))
    f2(sim, 'figures/talk_F2_sim_checks.png')
    f3(v3, 'figures/talk_F3_hidden_r.png')
    f4(v3, 'figures/talk_F4_paired.png')
    print('written figures/talk_F2_sim_checks.png, talk_F3_hidden_r.png, talk_F4_paired.png')
