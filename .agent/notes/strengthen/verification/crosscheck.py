"""Cross-check paper macros (tables/num_strengthen.tex) and the built PDF text against the exported CSVs.

Reads only eval_krea/strengthen/analysis_*.csv (exports of the analysis) and the macro file; formats each CSV value
with the paper's precision and compares strings. Usage: crosscheck.py <num_strengthen.tex> <text dump of main.pdf, or -> <out.json>
"""
import json, re, sys
import pandas as pd

X = '/scratch/09032/saini_2/constancy_rope/eval_krea/strengthen/'
mac = dict(re.findall(r'\\csname N@([^\\]+)\\endcsname\{(.*)\}$', open(sys.argv[1]).read(), re.M))


def f(v, d, sign=True):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if s.lstrip('+-').strip('0.') == '':
        s = ('-' if v < 0 else ('+' if sign else '')) + f"{0:.{d}f}"
    return s


def strip(m):
    return m.replace('$', '').replace('^{*}', '')


checks = []


def chk(name, macro, expect):
    got = strip(mac.get(macro, 'MISSING'))
    checks.append({'check': name, 'macro': macro, 'macro_value': got, 'csv_value': expect, 'ok': got == expect})


def row(frame, **kw):
    sel = frame
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, (kw, len(sel))
    return sel.iloc[0]


fc = pd.read_csv(X + 'analysis_factorial_contrasts_holm.csv'); fc = fc[fc.analysis == 'seed0']
for c, k in [('temperature_main_effect', 'temp'), ('base_main_effect', 'base'), ('interaction', 'inter'), ('allocation', 'alloc')]:
    for m, mk, d in [('imagereward', 'ir', 3), ('musiq', 'musiq', 2), ('pickscore', 'pick', 2)]:
        r = row(fc, contrast=c, metric=m)
        chk(f'factorial {c} {m} delta', f'st@f@d@{k}@{mk}', f(r.delta, d))
        chk(f'factorial {c} {m} CI', f'st@f@ci@{k}@{mk}', f'[{f(r.lo, d)}, {f(r.hi, d)}]')
chk('factorial Holm rejections', 'st@f@nholm', str(int(fc.holm_reject.sum())))
fam = pd.read_csv(X + 'analysis_fresh_confirmation_family.csv').set_index('hypothesis')
for h, k, d in [('H1_IR', 'ir', 3), ('H1_Pick', 'pick', 3), ('H2_MUSIQ', 'musiq', 2)]:
    r = fam.loc[h]
    chk(f'{h} delta', f'st@h@d@{k}', f(r.delta, d))
    chk(f'{h} CI', f'st@h@ci@{k}', f'[{f(r.lo, d)}, {f(r.hi, d)}]')
    chk(f'{h} Holm p', f'st@h@p@{k}', f'{r.p_holm:.2g}')
    chk(f'{h} status', f'st@h@status@{k}', r.status.replace('_', ' '))
ec = pd.read_csv(X + 'analysis_fresh_contrasts_holm.csv'); ec = ec[ec.analysis == 'seed0']
for a, k in [('dype', 'dype'), ('ntk2_sref', 'ntktemp'), ('temp_only_sbase', 'tsb')]:
    for m, mk, d in [('imagereward', 'ir', 3), ('musiq_crop1024', 'musiqc', 2)]:
        r = row(ec, contrast=a, metric=m)
        chk(f'fresh {a} {m}', f'st@e@d@{k}@{mk}', f(r.delta, d))
sc = pd.read_csv(X + 'analysis_seeds_contrasts_holm.csv')
for lab in ['seeds012', 'seeds12']:
    for m, mk, d in [('imagereward', 'ir', 3), ('musiq', 'musiq', 2)]:
        r = row(sc, analysis=lab, contrast='temp_only_sbase', metric=m)
        chk(f'seeds {lab} {m}', f'st@r{lab}@d@tsb@{mk}', f(r.delta, d))
        chk(f'seeds {lab} {m} CI', f'st@r{lab}@ci@tsb@{mk}', f'[{f(r.lo, d)}, {f(r.hi, d)}]')
cv = pd.read_csv(X + 'analysis_tsweep_curve.csv')
for tau, k in [(1.0, 't100'), (1.1, 't110'), (4 / 3, 'tlogn'), (1.45, 't145')]:
    for m, mk, d in [('imagereward', 'ir', 3), ('musiq_crop1024', 'musiqc', 2)]:
        r = cv[(abs(cv.logit_multiplier - tau) < 1e-9) & (cv.metric == m)].iloc[0]
        chk(f'sweep mean {tau} {m}', f'st@sm@{k}@{mk}', f(r['mean'], d, False))
ds = pd.read_csv(X + 'analysis_default_shift_paired.csv')
for m, mk, d in [('imagereward', 'ir', 3), ('musiq', 'musiq', 2)]:
    r = row(ds, metric=m)
    chk(f'default shift {m}', f'st@ds@d@def@{mk}', f(r['mean'], d))
    chk(f'default shift {m} CI', f'st@ds@ci@def@{mk}', f'[{f(r.ci_low, d)}, {f(r.ci_high, d)}]')
ls = pd.read_csv(X + 'analysis_ckpt_matrix_lattice_summary.csv')
for g, k in [(3.5, 'g3p5'), (4.5, 'g4p5')]:
    r = row(ls, guidance=g)
    chk(f'lattice ratio g{g}', f'st@k@ratio@{k}', f(r['mean'], 1, False))
    chk(f'lattice ratio g{g} CI', f'st@k@ci@{k}', f'[{f(r.ci_low, 1, False)}, {f(r.ci_high, 1, False)}]')
l2 = pd.read_csv(X + 'analysis_ckpt_matrix_lattice2x2_summary.csv').set_index('quantity')
for q, k in [('guidance_g4p5_over_g3p5_dev', 'gdev'), ('guidance_g4p5_over_g3p5_krea', 'gkrea')]:
    r = l2.loc[q]
    chk(f'lattice {q}', f'st@k@two@{k}', f(r.geometric_mean, 2, False))
    chk(f'lattice {q} CI', f'st@k@twoci@{k}', f'[{f(r.ci_low, 2, False)}, {f(r.ci_high, 2, False)}]')
pdf_hits = {}
if sys.argv[2] != '-':
    text = open(sys.argv[2]).read().replace('−', '-')
    for c in checks:
        if c['ok'] and len(c['csv_value']) >= 4:
            pdf_hits[c['macro']] = c['csv_value'] in text
out = {'n_checks': len(checks), 'n_ok': sum(c['ok'] for c in checks), 'failures': [c for c in checks if not c['ok']],
       'pdf_text_found': sum(pdf_hits.values()), 'pdf_text_checked': len(pdf_hits),
       'pdf_text_missing': [k for k, v in pdf_hits.items() if not v], 'checks': checks}
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'checks'})
