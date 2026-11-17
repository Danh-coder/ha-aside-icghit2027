"""Frozen, nested whole-group ChatQA subsets and a source-only development set."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from experiments.common import digest, file_hash, write_json
from tools.prepare_chatqa_dataset import load_prepared

PROTOCOL='ha_chatqa_multifidelity_hpo_v1'
SEED=42
UNKNOWN='Sorry. I cannot find the answer based on the context.'


def key(text):
    text=' '.join(text.casefold().split())
    return hashlib.sha256(text.encode()).hexdigest() if len(text)>=40 and text!=UNKNOWN.casefold() else None


def fields(row):
    return [row['document'],*[m['content'] for m in row['messages']],*row['answers']]


def benchmark_exclusions(directory):
    """Source/gold exact-field exclusion only, with full file fingerprints."""
    hashes={};forbidden=set()
    def strings(value):
        if isinstance(value,str):yield value
        elif isinstance(value,list):
            for v in value:yield from strings(v)
        elif isinstance(value,dict):
            for v in value.values():yield from strings(v)
    for suite in ['role_authority','task_controls','iheval']:
        path=Path(directory)/(suite+'.jsonl');hashes[suite]=file_hash(path)
        with path.open(encoding='utf-8') as stream:
            for case in map(json.loads,stream):
                for msg in case['messages']:
                    for part in msg['segments']:
                        value=key(part['text']) if not part.get('framing') else None
                        if value:forbidden.add(value)
                for text in strings(case['scorer'].get('answer',case['scorer'].get('answers',[]))):
                    value=key(text)
                    if value:forbidden.add(value)
    return forbidden,hashes


def freeze(splits, forbidden=()):
    groups=defaultdict(list)
    for index,row in enumerate(splits['train']):groups[row['group_id']].append(index)
    buckets=defaultdict(list)
    for group,indices in groups.items():
        rows=[splits['train'][i] for i in indices]
        largest=max(r['n_tokens'] for r in rows)
        length=sum(largest>limit for limit in [512,1024,1536])
        multi=any(len(r['messages'])>1 for r in rows)
        unknown=any(r['answers'][0]==UNKNOWN for r in rows)
        buckets[f'length{length}_multi{int(multi)}_unknown{int(unknown)}'].append(group)
    ordered={name:sorted(ids,key=lambda g:digest([PROTOCOL,SEED,'train',g])) for name,ids in sorted(buckets.items())}
    subsets={}
    for stage,fraction in [('screen',.1),('refine',.3),('full',1.)]:
        chosen={g for ids in ordered.values() for g in ids[:math.ceil(len(ids)*fraction)]}
        indices=[i for i,r in enumerate(splits['train']) if r['group_id'] in chosen]
        rows=[splits['train'][i] for i in indices]
        subsets[stage]={'group_fraction_target':fraction,'group_count':len(chosen),'row_indices':indices,
                        'groups':sorted(chosen),'rows':len(rows),'tokens_per_epoch':sum(r['n_tokens'] for r in rows),
                        'row_identity_sha256':digest([r['source_row_sha256'] for r in rows]),
                        'strata':{name:math.ceil(len(ids)*fraction) for name,ids in ordered.items()}}
    validation=defaultdict(list);blocked=set()
    for index,row in enumerate(splits['validation']):
        validation[row['group_id']].append((index,row))
        if any(key(text) in forbidden for text in fields(row) if key(text)):blocked.add(row['group_id'])
    cases=[]
    for group in sorted(set(validation)-blocked,key=lambda g:digest([PROTOCOL,SEED,'dev-group',g])):
        line,row=min(validation[group],key=lambda pair:digest([PROTOCOL,SEED,'dev-row',pair[1]['source_row_sha256']]))
        cases.append({'case_id':'hpo_dev:'+str(row['source_row'])+':'+row['source_row_sha256'][:12],
                      'validation_line':line+1,'row':row,
                      'source':{'original_source_row':row['source_row'],'source_row_sha256':row['source_row_sha256']}})
    if not cases:raise ValueError('No eligible validation groups remain for HPO dev')
    return subsets,cases,{'train_groups':len(groups),'validation_groups':len(validation),
                         'dev_excluded_groups_exact_RQ_overlap':len(blocked),
                         'dev_groups':len(cases),'train_strata_sizes':{k:len(v) for k,v in ordered.items()},
                         'selection':'ceil(fraction * groups) within length/multi-turn/unanswerable strata; fixed seeded group order',
                         'dev':'one unchanged validation row per eligible group; all eligible groups, not 88-row pilot',
                         'overlap_scope':'exact normalized fields >=40 chars; not semantic/near-duplicate certification'}


def prepare(data_dir, output, rq_dir):
    manifest,splits=load_prepared(data_dir)
    forbidden,rq_hashes=benchmark_exclusions(rq_dir)
    subsets,cases,audit=freeze(splits,forbidden)
    payload=''.join(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n' for c in cases).encode()
    identity={'protocol':PROTOCOL,'seed':SEED,'data_manifest_canonical_sha256':digest(manifest),
              'train_sha256':manifest['files']['train.jsonl']['sha256'],
              'validation_sha256':manifest['files']['validation.jsonl']['sha256'],
              'public_source':{k:manifest[k] for k in ['source_repo','source_revision','source_sha256']},
              'subsets':subsets,'dev_cases_sha256':hashlib.sha256(payload).hexdigest(),
              'RQ_exclusion_sha256':rq_hashes,'audit':audit,'source_strings_rewritten':False,
              'subset_fractions_are_group_based':True,'selection_uses_model_outputs':False}
    identity['fingerprint']=digest(identity)
    output=Path(output)
    if output.exists():
        if json.loads((output/'manifest.json').read_text(encoding='utf-8'))!=identity or (output/'dev_cases.jsonl').read_bytes()!=payload:
            raise FileExistsError('Frozen HPO data differ; choose a new prepared directory')
    else:
        output.mkdir(parents=True)
        (output/'dev_cases.jsonl').write_bytes(payload);write_json(output/'manifest.json',identity)
    return identity


def load(prepared, data_dir):
    prepared=Path(prepared)
    frozen=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if digest({k:v for k,v in frozen.items() if k!='fingerprint'})!=frozen['fingerprint']:
        raise ValueError('HPO data selection changed')
    if file_hash(prepared/'dev_cases.jsonl')!=frozen['dev_cases_sha256']:
        raise ValueError('HPO dev cases changed')
    manifest,splits=load_prepared(data_dir)
    if digest(manifest)!=frozen['data_manifest_canonical_sha256']:
        raise ValueError('Full source corpus differs from the frozen HPO corpus')
    selected={}
    for stage,item in frozen['subsets'].items():
        rows=[splits['train'][i] for i in item['row_indices']]
        if digest([r['source_row_sha256'] for r in rows])!=item['row_identity_sha256']:
            raise ValueError('Subset content differs')
        if sorted({r['group_id'] for r in rows})!=item['groups'] or len(rows)!=item['rows']:
            raise ValueError('Subset group identity differs')
        chosen=set(item['groups'])
        if item['row_indices']!=[i for i,r in enumerate(splits['train']) if r['group_id'] in chosen]:
            raise ValueError('A subset has dropped part of a source group')
        selected[stage]=rows
    if not set(frozen['subsets']['screen']['groups'])<=set(frozen['subsets']['refine']['groups'])<=set(frozen['subsets']['full']['groups']):
        raise ValueError('Train subsets must be nested')
    with (prepared/'dev_cases.jsonl').open(encoding='utf-8') as stream:cases=[json.loads(line) for line in stream]
    if len({c['row']['group_id'] for c in cases})!=len(cases):raise ValueError('Dev source group duplicated')
    train_groups={r['group_id'] for r in splits['train']}
    for case in cases:
        if case['row']!=splits['validation'][case['validation_line']-1] or case['row']['group_id'] in train_groups:
            raise ValueError('Dev source/gold modified or leaked into train')
    return frozen,manifest,splits,selected,cases
