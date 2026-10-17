"""Tabulate the fixed CR20 rubric without resolving reviewer disagreements."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / '.agent/notes/native_cr20'
LABELS = ('satisfied', 'violated', 'unjudgeable')
COMMON_HASHES = {
    'a': 'da2ef7ee92d869fac80d8b73e90190bc1817987328c9ecb3d6b0548ca31e20c0',
    'b': 'cb0b8bed61b6f1220ee2f72255de5a95331ba1b184ce90703bacdb7c97ce14bc',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((NOTES / 'report.json').read_text())
    mapping = {v['label']: v for v in report['viewmap']}
    reviews, hashes = {}, {}
    for reviewer in ('a', 'b'):
        common_path = NOTES / f'reviewer_{reviewer}_common.json'
        assert sha(common_path) == COMMON_HASHES[reviewer], 'Common labels changed after freeze'
        for stage in ('common', 'detail'):
            path = NOTES / f'reviewer_{reviewer}_{stage}.json'
            hashes[path.name] = sha(path)
            raw = json.loads(path.read_text())
            entries = {v['label']: v for v in raw['images']}
            assert len(raw['images']) == len(entries) == 16 and set(entries) == set(mapping)
            for label, item in entries.items():
                for role in ('primary', 'secondary'):
                    assert item[role]['label'] in LABELS and item[role]['evidence']
                assert all(x['label'] in LABELS and x['evidence'] for x in item['auxiliary'])
                if stage == 'common':
                    assert item['sha256'] == mapping[label]['view_sha256']
                    assert sha(Path(item['inspected_image_path'])) == item['sha256']
            rows = []
            matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
            for index in range(8):
                pair = {v['condition']: entries[label] for label, v in mapping.items() if v['row'] == index}
                assert set(pair) == {'native1024', 'archived4096'}
                matrix[pair['native1024']['primary']['label']][pair['archived4096']['primary']['label']] += 1
                rows.append({'row': index, **{condition: {
                    'view': entry['label'], **{role: entry[role] for role in ('primary', 'secondary', 'auxiliary')}
                } for condition, entry in pair.items()}})
            reviews[f'{reviewer}_{stage}'] = {'rows': rows, 'primary_native_by_4k': matrix}
    disagreements, changes = [], []
    for stage in ('common', 'detail'):
        for index, (a, b) in enumerate(zip(reviews[f'a_{stage}']['rows'], reviews[f'b_{stage}']['rows'])):
            for condition in ('native1024', 'archived4096'):
                for role in ('primary', 'secondary', 'auxiliary'):
                    def labels(row):
                        entry = row[condition][role]
                        return [x['label'] for x in entry] if role == 'auxiliary' else entry['label']
                    if labels(a) != labels(b):
                        disagreements.append(dict(stage=stage, row=index, condition=condition, role=role))
                    for reviewer in ('a', 'b'):
                        if stage == 'detail' and labels(reviews[f'{reviewer}_common']['rows'][index]) != labels(reviews[f'{reviewer}_detail']['rows'][index]):
                            changes.append(dict(reviewer=reviewer, row=index, condition=condition, role=role))
    result = {
        'scope': 'Two independent model-based rubric readings of eight observational development pairs; no human study or causal resolution claim.',
        'reviewer_models': {'a': 'GPT-6 Astra high', 'b': 'GPT-6.1 Sol high'},
        'source_sha256': {**hashes, 'report.json': sha(NOTES / 'report.json'), 'summary_script': sha(Path(__file__))},
        'reviews': reviews, 'reviewer_label_disagreements': disagreements, 'common_to_detail_changes': changes,
        'detail_policy': 'Five originals per reviewer inspected for ambiguous/violated predicates and auxiliary text; eleven clear views carried forward explicitly. Detail is not a second full-image study.',
        'decision': 'No primary native-satisfied/4K-violated example. Close this particular semantic-loss rationale; no seed search, OCR pivot, fresh64 or method promotion follows.',
        'limits': ['One seed, eight previously inspected development captions; no failure-rate estimate.',
                   'Noise shapes are different and uncoupled; deployed RoPE and gains also change with resolution.',
                   'Label-hidden presentation does not guarantee resolution blinding or independent model errors.',
                   'No inference that other resolution failures are absent.'],
    }
    (NOTES / 'visual_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'primary_matrices': {k: v['primary_native_by_4k'] for k, v in reviews.items()},
                      'disagreements': disagreements, 'detail_changes': changes}))


if __name__ == '__main__':
    main()
