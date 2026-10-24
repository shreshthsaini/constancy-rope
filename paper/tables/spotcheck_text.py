"""Spot-check: recompute headline numbers from .agent/notes/strengthen/analysis and confirm the compiled main text
(pages 1-10 of paper/main.pdf) prints them. Run with the pdfrender env:
PYTHONPATH=/work/09032/saini_2/vista/library/uv-envs/pdfrender/lib/python3.11/site-packages python3.11 -s paper/tables/spotcheck_text.py
"""
# Spot-check: recompute numbers from per-image / analysis CSVs and check the compiled main text (pages 1-10) prints them.
import csv, json, re, statistics, sys, collections
import fitz
R='/work/09032/saini_2/vista/Projects/Exploration/constancy_rope/'
A=R+'.agent/notes/strengthen/analysis/'
d=fitz.open(R+'paper/main.pdf')
txt=' '.join(d[i].get_text() for i in range(10)).replace('−','-').replace('\n',' ')
txt=re.sub(r'\s+',' ',txt)
def rows(p): return list(csv.DictReader(open(p)))
def f(x,k,sign=True): s=f'{x:+.{k}f}' if sign else f'{x:.{k}f}'; return s
out=[]
def check(name, computed, printed):
    ok = printed in txt
    out.append((name, computed, printed, ok))
# --- P1 per-image recomputation (seed 0, held-out rows) ---
pi=rows(A+'P1/canonical_per_image.csv')
hdr=list(pi[0].keys())
irkey=[k for k in hdr if 'imagereward' in k.lower()]
by=collections.defaultdict(dict)
for r in pi:
    if r.get('seed','0') not in ('0','0.0'): continue
    by[r['method']][r['prompt_id']]=r
arms=sorted(by)
def mean(arm,m): v=[float(r[m]) for r in by[arm].values() if r.get(m) not in (None,'')]; return sum(v)/len(v), len(v)
def paired(a,b,m):  # a minus b over common prompts
    ps=sorted(set(by[a])&set(by[b])); return sum(float(by[a][p][m])-float(by[b][p][m]) for p in ps)/len(ps), len(ps)
print('arms', arms, 'ir key', irkey)
ch=rows(A+'P1/contrasts_holm.csv')
def C(contrast, metric):
    r=[x for x in ch if x['analysis']=='seed0' and x['contrast']==contrast and x['metric']==metric][0]
    return float(r['delta']), float(r['lo']), float(r['hi']), r['holm_reject']
def ci(lo,hi,k): return f'[{lo:+.{k}f}, {hi:+.{k}f}]'
# 1-2 factorial cells and total, recomputed from per-image rows
for arm,lab in [('ntk2','NTK'),('ntk2_sref','NTK+temp'),('sega_base_none','SEGA base'),('temp_only_sbase','TSB')]:
    m,n=mean(arm,'imagereward'); check(f'P1 mean IR {lab} (n={n})', round(m,3), f'{m:.3f}')
tot,n=paired('sega_official','ntk2','imagereward'); check(f'total SEGA-NTK IR (n={n})', round(tot,3), f'{tot:+.3f}')
totm,n=paired('sega_official','ntk2','musiq'); check('total SEGA-NTK full MUSIQ', round(totm,2), f'{totm:+.2f}')
# temperature main effect recomputed: mean of the two simple effects
t1,_=paired('ntk2_sref','ntk2','imagereward'); t2,_=paired('temp_only_sbase','sega_base_none','imagereward')
dd,lo,hi,h=C('temperature_main_effect','imagereward'); check('temp main IR point (recomputed)', round((t1+t2)/2,3), f'{(t1+t2)/2:+.3f}')
check('temp main IR CI', ci(lo,hi,3), ci(lo,hi,3))
check('temp at NTK base IR', round(t1,3), f'{t1:+.3f}'); check('temp at SEGA base IR', round(t2,3), f'{t2:+.3f}')
b1,_=paired('sega_base_none','ntk2','imagereward'); b2,_=paired('temp_only_sbase','ntk2_sref','imagereward')
dd,lo,hi,h=C('base_main_effect','imagereward'); check('base main IR point+CI', round((b1+b2)/2,3), f'{(b1+b2)/2:+.3f} {ci(lo,hi,3)}')
tm1,_=paired('ntk2_sref','ntk2','musiq'); tm2,_=paired('temp_only_sbase','sega_base_none','musiq')
dd,lo,hi,h=C('temperature_main_effect','musiq'); check('temp main full MUSIQ point+CI', round((tm1+tm2)/2,2), f'{(tm1+tm2)/2:+.2f}')
check('temp main full MUSIQ CI', ci(lo,hi,2), ci(lo,hi,2))
bc1,_=paired('sega_base_none','ntk2','musiq_crop1024'); bc2,_=paired('temp_only_sbase','ntk2_sref','musiq_crop1024')
dd,lo,hi,h=C('base_main_effect','musiq_crop1024'); check(f'base main crop MUSIQ (Holm {h})', round((bc1+bc2)/2,2), f'{(bc1+bc2)/2:+.2f} {ci(lo,hi,2)}')
al,_=paired('sega_official','temp_only_sbase','musiq'); dd,lo,hi,h=C('allocation','musiq'); check(f'allocation full MUSIQ (Holm {h})', round(al,2), f'{al:+.2f}')
check('allocation full MUSIQ CI', ci(lo,hi,2), ci(lo,hi,2))
ai,_=paired('sega_official','temp_only_sbase','imagereward'); dd,lo,hi,h=C('allocation','imagereward'); check('allocation IR point+CI', round(ai,3), f'{ai:+.3f} {ci(lo,hi,3)}')
nh=sum(1 for x in ch if x['analysis']=='seed0' and x['holm_reject']=='True'); check('P1 Holm survivors', nh, f'{nh} survive Holm')
# registered family
fam=json.load(open(A+'E1/confirmation_family.json'))['family']
for r in fam:
    k=2 if r['hypothesis']=='H2_MUSIQ' else 3
    check(f"{r['hypothesis']} n={r['n']} Holm p {r['p_holm']:.2g}", round(r['delta'],k), f"{r['delta']:+.{k}f} {ci(r['lo'],r['hi'],k)}")
hy=json.load(open(A+'E1/hypotheses.json'))
fp=[r for r in hy['seed0']['family'] if r['hypothesis']=='H1_Pick'][0]; check(f"fresh-only PickScore ({fp['status']})", round(fp['delta'],3), f"{fp['delta']:+.3f} {ci(fp['lo'],fp['hi'],3)}")
# fresh per-image means recomputed
pe=rows(A+'E1/canonical_per_image.csv'); fr=collections.defaultdict(list)
for r in pe:
    if r['seed'] in ('0','0.0'): fr[(r['method'],'pickscore')].append(float(r['pickscore'])); fr[(r['method'],'musiq')].append(float(r['musiq']))
for arm in ('sega_official','temp_only_sbase'):
    v=fr[(arm,'pickscore')]; check(f'fresh mean PickScore {arm} n={len(v)}', round(sum(v)/len(v),2), f'{sum(v)/len(v):.2f}')
    v=fr[(arm,'musiq')]; check(f'fresh mean MUSIQ {arm} n={len(v)}', round(sum(v)/len(v),2), f'{sum(v)/len(v):.2f}')
# lattice and default shift
# shared-bin lattice ratios (2026-10-24: the text quotes the shared bin, not the upward-biased historical peak bin)
lb={r['quantity']: r for r in rows(A+'ckpt_matrix_lattice2x2/summary.csv')}
g35=lb['checkpoint_dev_over_krea_at_g3p5']; g45=lb['checkpoint_dev_over_krea_at_g4p5']
check('shared-bin lattice ratio g3.5 (n=%s)' % g35['n'], round(float(g35['geometric_mean']),1),
      f"{float(g35['geometric_mean']):.1f} times [{float(g35['ci_low']):.1f}, {float(g35['ci_high']):.1f}]")
check('shared-bin lattice ratio g4.5 (n=%s)' % g45['n'], round(float(g45['geometric_mean']),1),
      f"{float(g45['geometric_mean']):.1f}\u00d7 [{float(g45['ci_low']):.1f}, {float(g45['ci_high']):.1f}]")
ds=[r for r in rows(A+'default_shift/paired.csv') if r['metric']=='imagereward'][0]
check('default shift IR', round(float(ds['mean']),3), f"{float(ds['mean']):+.3f} {ci(float(ds['ci_low']),float(ds['ci_high']),3)}")
# --- additions 2026-10-24 (text agent, gate review v3 fixes) ---
nospace = txt.replace(' ', '')
def check_ns(name, computed, printed):
    out.append((name, computed, printed, printed.replace(' ', '') in nospace))
# allocation's fresh PickScore gain: released SEGA minus base + temp., recomputed per prompt from E1 per-image rows
fp_ = collections.defaultdict(dict)
for r in pe:
    if r['seed'] in ('0','0.0'): fp_[r['method']][r['prompt_id']] = float(r['pickscore'])
ps_ = sorted(set(fp_['sega_official']) & set(fp_['temp_only_sbase']))
ga = sum(fp_['sega_official'][q] - fp_['temp_only_sbase'][q] for q in ps_) / len(ps_)
ec = [x for x in rows(A+'E1/contrasts_holm.csv') if x['analysis']=='seed0' and x['contrast']=='temp_only_sbase' and x['metric']=='pickscore'][0]
check(f'fresh allocation PickScore (n={len(ps_)}), CI from E1 bootstrap', round(ga,3), f"{ga:+.3f} {ci(-float(ec['hi']), -float(ec['lo']), 3)}")
check('fresh allocation PickScore Holm p (family of 50)', round(float(ec['p_holm']),2), f"Holm \U0001d45d= {float(ec['p_holm']):.2f} over 50 tests")
# Table 3 total row: paired SEGA minus NTK interval, from P1/paired.csv (sign reversed)
pr_ = [x for x in rows(A+'P1/paired.csv') if x['analysis']=='seed0' and x['method']=='ntk2' and x['reference']=='sega_official' and x['metric']=='imagereward'][0]
check_ns('Table 3 total IR interval', ci(-float(pr_['hi']), -float(pr_['lo']), 3), ci(-float(pr_['hi']), -float(pr_['lo']), 3))
# sweep: 1.45 beats s^0.16 on crop MUSIQ, recomputed per prompt from P3 per-image rows
sw = collections.defaultdict(dict)
for r in rows(A+'P3/canonical_per_image.csv'):
    if r['seed'] in ('0','0.0'): sw[r['method']][r['prompt_id']] = float(r['musiq_crop1024'])
q3 = sorted(set(sw['tau1p45']) & set(sw['tau_sref']))
d145 = sum(sw['tau1p45'][q] - sw['tau_sref'][q] for q in q3) / len(q3)
check(f'sweep 1.45 minus s^0.16 crop MUSIQ (n={len(q3)})', round(d145,2), f'{d145:+.2f}')
# amendment disclosure: fresh-only ImageReward interval against the original +/-0.05 rule
fi = [r for r in hy['seed0']['family'] if r['hypothesis']=='H1_IR'][0]
check(f"fresh-only ImageReward interval (lo {fi['lo']:.4f} < -0.05)", ci(fi['lo'], fi['hi'], 3), ci(fi['lo'], fi['hi'], 3))
# shared-bin guidance scaling on FLUX.1-dev
gd = lb['guidance_g4p5_over_g3p5_dev']
check('guidance scaling of the FLUX.1-dev lattice', round(float(gd['geometric_mean']),2), f"{float(gd['geometric_mean']):.2f}\u00d7 [{float(gd['ci_low']):.2f}, {float(gd['ci_high']):.2f}]")
for name,comp,pr,ok in out: print(('OK  ' if ok else 'MISS'), name, '|', comp, '|', pr)
print(sum(o[3] for o in out), 'of', len(out), 'found in main text')
