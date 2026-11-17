"""Frozen, resumable ChatQA-validation utility diagnostic for saved HA epochs.

Additive handoff: does not edit old checkpoints, shared generators or RQ results.
Only public source rows/questions/answers are used; no generated benchmark data.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import ROOT as WORKSPACE, digest, file_hash, write_json

PROTOCOL = 'ha_epoch_chatqa_dev_v1'
PREPARED = ROOT / 'data/prepared/ha_epoch_dev_v1'
STEPS = [501, 1002, 1503]
STOPS = [151643, 151645]
MAX_NEW = 1024
MAX_CONTEXT = 16384
CANONICAL_UNKNOWN = 'Sorry. I cannot find the answer based on the context.'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(text):
    return ' '.join(text.casefold().split())


def field_key(text):
    text = normalized(text)
    return hashlib.sha256(text.encode()).hexdigest() if len(text) >= 40 and text != normalized(CANONICAL_UNKNOWN) else None


def origin_fields(row):
    return [row['document'], *[m['content'] for m in row['messages']], *row['answers']]


def choose_rows(validation, forbidden, fraction=.1):
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    groups = defaultdict(list)
    blocked = set()
    for index, row in enumerate(validation):
        original = {k: row[k] for k in ['document', 'messages', 'answers']}
        if digest(original) != row['source_row_sha256']:
            raise ValueError('Modified validation source row')
        groups[row['group_id']].append((index, row))
        if any(field_key(text) in forbidden for text in origin_fields(row) if field_key(text)):
            blocked.add(row['group_id'])
    eligible = sorted(set(groups) - blocked, key=lambda g: digest([PROTOCOL, 42, 'group', g]))
    if not eligible:
        raise ValueError('No validation groups remain after exact-overlap exclusion')
    chosen = eligible[:math.ceil(len(eligible)*fraction)]
    selected = [min(groups[g], key=lambda pair: digest([PROTOCOL, 42, 'row', pair[1]['source_row_sha256']])) for g in chosen]
    assert len({row['group_id'] for _, row in selected}) == len(selected)
    return selected, {'validation_rows': len(validation), 'validation_groups': len(groups),
                      'excluded_groups_exact_field_overlap': len(blocked), 'eligible_groups': len(eligible),
                      'fraction_of_eligible_groups': fraction, 'selected_groups': len(selected),
                      'selected_rows': len(selected), 'seed': 42,
                      'selection': 'one unchanged row per source group; deterministic source hashes before inference',
                      'overlap_scope': 'exact whitespace/case-normalized source fields >=40 chars; not a semantic/near-duplicate guarantee'}


def prepare(data_dir, output):
    from tools.prepare_chatqa_dataset import load_prepared
    manifest, splits = load_prepared(data_dir)
    forbidden, exclusions = set(), {}
    source_dir = ROOT / 'data/prepared/rq_public_v1'
    for suite in ['role_authority', 'task_controls', 'iheval']:
        path = source_dir / (suite+'.jsonl')
        exclusions[path.relative_to(WORKSPACE).as_posix()] = file_hash(path)
        for row in read_rows(path):
            for msg in row['messages']:
                for part in msg['segments']:
                    key = field_key(part['text']) if not part.get('framing') else None
                    if key:
                        forbidden.add(key)
            def strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, list):
                    for v in value:
                        yield from strings(v)
                elif isinstance(value, dict):
                    for v in value.values():
                        yield from strings(v)
            for text in strings(row['scorer'].get('answer', row['scorer'].get('answers', []))):
                key = field_key(text)
                if key:
                    forbidden.add(key)
    selected, audit = choose_rows(splits['validation'], forbidden)
    cases = []
    for index, row in selected:
        cases.append({'case_id': 'chatqa_val:'+str(row['source_row'])+':'+row['source_row_sha256'][:12],
                      'validation_line': index+1, 'row': row,
                      'source': {'source_repo': manifest['source_repo'], 'source_revision': manifest['source_revision'],
                                 'original_source_row': row['source_row'], 'source_row_sha256': row['source_row_sha256']}})
    payload = ''.join(json.dumps(c, ensure_ascii=False, allow_nan=False)+'\n' for c in cases).encode()
    identity = {'protocol': PROTOCOL, 'purpose': 'development QA utility diagnosis; not RQ benchmark results',
                'training_data_manifest_canonical_sha256': digest(manifest),
                'validation_sha256': manifest['files']['validation.jsonl']['sha256'],
                'train_sha256': manifest['files']['train.jsonl']['sha256'],
                'public_source': {k: manifest[k] for k in ['source_repo','source_revision','source_sha256','source_url']},
                'excluded_RQ_source_files': exclusions, 'forbidden_fields_sha256': digest(sorted(forbidden)),
                'audit': audit, 'cases_sha256': hashlib.sha256(payload).hexdigest(),
                'max_new_tokens': MAX_NEW, 'max_context': MAX_CONTEXT, 'stop_ids': STOPS,
                'metrics': ['normalized_exact_match','token_f1','rougeL'],
                'history': 'original reference history; no own-history rollout',
                'selection_uses_model_outputs': False}
    identity['fingerprint'] = digest(identity)
    output = Path(output)
    if output.exists():
        if read_json(output/'manifest.json') != identity or (output/'cases.jsonl').read_bytes() != payload:
            raise FileExistsError('Dev cohort already exists with different identity; choose a new output directory')
    else:
        output.mkdir(parents=True)
        (output/'cases.jsonl').write_bytes(payload)
        write_json(output/'manifest.json', identity)
    print(json.dumps(identity, ensure_ascii=False, indent=2))


def cohort(folder):
    folder = Path(folder)
    info = read_json(folder/'manifest.json')
    unsigned = {k: v for k,v in info.items() if k != 'fingerprint'}
    if info['protocol'] != PROTOCOL or digest(unsigned) != info['fingerprint']:
        raise ValueError('Dev manifest identity changed')
    if file_hash(folder/'cases.jsonl') != info['cases_sha256']:
        raise ValueError('Frozen dev cases changed')
    cases = read_rows(folder/'cases.jsonl')
    if len(cases) != info['audit']['selected_rows'] or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Dev case count or IDs differ')
    if len({c['row']['group_id'] for c in cases}) != len(cases):
        raise ValueError('Dev rows are not group-disjoint')
    for c in cases:
        row = c['row']
        if digest({k: row[k] for k in ['document','messages','answers']}) != row['source_row_sha256']:
            raise ValueError('Dev question/document/gold changed')
    return info, cases


def checkpoint_evidence(path, step, dev):
    from experiments.checkpoints import checkpoint_metadata
    meta = checkpoint_metadata(path, 'ha_aside', 'qwen25_7b')
    provenance = meta['provenance']
    expected = {'0': 0., '1': math.pi/8, '2': math.pi/2, '3': 0.}
    if meta['enable_rgag'] or {str(k):float(v) for k,v in meta['angles'].items()} != expected or meta['rotation_direction'] != 'right':
        raise ValueError('Requires main HA angles, right rotation and RGAG OFF')
    if provenance.get('pilot_max_steps') is not None:
        raise ValueError('This diagnostic requires saved epochs from the full trial, not a bounded training pilot')
    if (provenance['training_recipe'] != 'chatqa_multirole_fullseq_2048_v1'
            or provenance['training_data_sha256'] != dev['train_sha256']
            or provenance['validation_data_sha256'] != dev['validation_sha256']
            or provenance['dataset_manifest_canonical_sha256'] != dev['training_data_manifest_canonical_sha256']):
        raise ValueError('Checkpoint training data differ from the frozen validation corpus')
    state = read_json(Path(path)/'trainer_state.json')
    if state['global_step'] != step or abs(state['epoch'] - step/501) > 1e-6 or state['max_steps'] != 1503:
        raise ValueError('Saved checkpoint step/epoch differ from this 1503-step trial')
    evaluations = [r for r in state['log_history'] if 'eval_loss' in r and r['step']==step]
    if not evaluations or abs(evaluations[-1]['eval_loss']-provenance['selected_validation_loss']) > 1e-9:
        raise ValueError('Missing matching Trainer validation evidence')
    return meta, {'epoch': state['epoch'], 'step': step, 'planned_max_steps': state['max_steps'],
                  'training_validation_loss': evaluations[-1]['eval_loss'],
                  'interpretation': 'saved epoch from one full training trial; not a completed-search claim'}


def verify_validation(data_dir, dev, cases, *, full=False):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir/'manifest.json')
    if digest(manifest) != dev['training_data_manifest_canonical_sha256'] or file_hash(data_dir/'validation.jsonl') != dev['validation_sha256']:
        raise ValueError('Original ChatQA validation corpus differs from the frozen dev source')
    if full:
        from tools.prepare_chatqa_dataset import load_prepared
        _,splits = load_prepared(data_dir)
        validation = splits['validation']
    else:
        validation = read_rows(data_dir/'validation.jsonl')
    for case in cases:
        index = case['validation_line']-1
        if index < 0 or index >= len(validation) or case['row'] != validation[index]:
            raise ValueError('Dev document/question/gold differs from its original validation line')


def setup(args):
    dev, cases = cohort(args.prepared)
    verify_validation(args.data_dir,dev,cases,full=True)
    paths = {str(step): str((args.trial_dir/('checkpoint-'+str(step))).resolve()) for step in STEPS}
    origin = None
    for step,path in paths.items():
        meta, _ = checkpoint_evidence(Path(path), int(step), dev)
        current = {k:v for k,v in meta['provenance'].items() if k != 'selected_validation_loss'}
        if origin is not None and current != origin:
            raise ValueError('Epoch checkpoints do not have the same training provenance')
        origin = current
    config = {'protocol': PROTOCOL, 'prepared': str(args.prepared.resolve()), 'data_dir':str(args.data_dir.resolve()),'dev_fingerprint':dev['fingerprint'],
              'checkpoints':paths,'steps':STEPS,'max_new_tokens':MAX_NEW,'max_context':MAX_CONTEXT,
              'stop_ids':STOPS,'device':'cuda:0','dtype':'bfloat16','seed':42,
              'output_root':str((ROOT/'results/ha_epoch_dev_v1').resolve()),
              'selection': 'No automatic winner; compare dev QA utility and generation defects. Test data excluded.'}
    if args.config.exists() and read_json(args.config) != config:
        raise FileExistsError('Existing config differs; choose a new config filename')
    write_json(args.config, config)
    print('CONFIG='+str(args.config)); print('Checkpoints found: '+', '.join(paths))


def encode_case(tokenizer, case, max_context):
    from experiments.prompts import PREFACE
    from experiments.rq_transcript import segment, message, encode_transcript
    row = case['row']
    messages = [message('system', segment(PREFACE,'system',framing=True)),
                message('user', segment('Document:\n','system',framing=True), segment(row['document'],'data'),
                        segment('\n\nQuestion:\n','system',framing=True), segment(row['messages'][0]['content'],'user'))]
    for item in row['messages'][1:]:
        messages.append(message(item['role'], segment(item['content'],item['role'])))
    return encode_transcript(tokenizer,messages,max_context)


def source_identity():
    paths = [Path(__file__), *[ROOT/'experiments'/n for n in ['checkpoints.py','rq_transcript.py','prompts.py','metrics.py','rq_live_io.py','rq_week_runner.py']],
             *[ROOT/'ha_aside'/n for n in ['hf_adapters.py','core.py','rgag.py']]]
    return {str(p.relative_to(WORKSPACE)):file_hash(p) for p in paths}


def preflight(config):
    import torch
    from transformers import AutoTokenizer
    report = {'ready':False,'errors':[],'checkpoints':{},'protocol':PROTOCOL}
    try:
        if config['protocol'] != PROTOCOL or config['steps'] != STEPS or config['stop_ids'] != STOPS or config['max_new_tokens'] != MAX_NEW or config['max_context'] != MAX_CONTEXT:
            raise ValueError('Config differs from the locked dev protocol')
        dev,cases = cohort(config['prepared'])
        if config['dev_fingerprint'] != dev['fingerprint']:
            raise ValueError('Config dev fingerprint differs')
        verify_validation(config['data_dir'],dev,cases)
        if config['dtype'] != 'bfloat16' or config['seed'] != 42:
            raise ValueError('Dev protocol requires bfloat16 and seed 42')
        for relative, expected_sha in dev['excluded_RQ_source_files'].items():
            if file_hash(WORKSPACE/relative) != expected_sha:
                raise ValueError('RQ exclusion source changed: '+relative)
        origin,tokenizer_files = None,None
        for step in config['steps']:
            path = Path(config['checkpoints'][str(step)])
            meta,evidence = checkpoint_evidence(path,step,dev)
            current = {k:v for k,v in meta['provenance'].items() if k != 'selected_validation_loss'}
            if origin is not None and origin != current:
                raise ValueError('Training provenance differs across epochs')
            origin = current
            files = {}
            for file in sorted(path.iterdir()):
                if file.is_file() and (file.suffix in {'.json','.safetensors','.jinja','.txt','.model'}):
                    print('[HASH] checkpoint-'+str(step)+'/'+file.name,flush=True)
                    files[file.name] = file_hash(file)
            signature = {k:v for k,v in files.items() if k.startswith(('tokenizer','special_tokens','added_tokens','vocab','merges','chat_template'))}
            if tokenizer_files is not None and signature != tokenizer_files:
                raise ValueError('Tokenizer/template files differ across epochs')
            tokenizer_files = signature
            report['checkpoints'][str(step)] = {'path':str(path),'metadata':meta,'evidence':evidence,'files_sha256':files,
                                               'saved_config_eos':read_json(path/'config.json').get('eos_token_id'),
                                               'saved_generation_eos':read_json(path/'generation_config.json').get('eos_token_id')}
        tokenizer = AutoTokenizer.from_pretrained(config['checkpoints']['1503'],local_files_only=True,trust_remote_code=False)
        if tokenizer.convert_tokens_to_ids('<|endoftext|>') != STOPS[0] or tokenizer.convert_tokens_to_ids('<|im_end|>') != STOPS[1]:
            raise ValueError('Tokenizer stop token identities differ')
        budgets = {}
        for case in cases:
            encoded = encode_case(tokenizer,case,config['max_context'])
            n = int(encoded['input_ids'].shape[1])
            if n+config['max_new_tokens'] > config['max_context']:
                raise ValueError('Context overflow; no truncation: '+case['case_id'])
            budgets[case['case_id']] = {'prompt_tokens':n,'transcript_sha256':digest(encoded['rendered_prompt'])}
        report.update(dev_fingerprint=dev['fingerprint'],cases=len(cases),budgets=budgets,source_sha256=source_identity(),
                      environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                                   'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                   'packages':{n:importlib.metadata.version(n) for n in ['transformers','tokenizers','numpy','rouge-score','matplotlib']}})
        if config['device'] != 'cuda:0' or not torch.cuda.is_available():
            raise ValueError('GPU inference requires CUDA on the GPU server')
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    report['ready'] = not report['errors']
    return report


@contextmanager
def resolved_stops(model, stops):
    previous = model.config.eos_token_id
    model.config.eos_token_id = list(stops)
    try:
        yield
    finally:
        model.config.eos_token_id = previous


def score_answer(output, answers):
    from experiments.metrics import qa_scores
    from rouge_score.rouge_scorer import RougeScorer
    score = qa_scores(output,answers)
    score['rougeL'] = max(RougeScorer(['rougeL'],use_stemmer=False).score(a,output)['rougeL'].fmeasure for a in answers)
    return score


def cached_records(path, case_map, *, repair=False, step=None, signature=None):
    from experiments.rq_live_io import committed_rows
    values,_ = committed_rows(path,repair=repair)
    result = {}
    for row in values:
        cid = row['case_id']
        if cid in result or cid not in case_map or row['case_sha256'] != digest(case_map[cid]) or row['stop_ids'] != STOPS:
            raise ValueError('Cached output has duplicate IDs or changed source/gold/stop policy')
        if row.get('record_sha256') != digest({k:v for k,v in row.items() if k != 'record_sha256'}):
            raise ValueError('Cached record content/checksum changed')
        if step is not None and row.get('checkpoint_step') != step:
            raise ValueError('Cached output belongs to a different epoch')
        if signature is not None and row.get('checkpoint_signature') != signature:
            raise ValueError('Cached output belongs to different checkpoint weights or metadata')
        if len(row['output_ids']) != row['generated_tokens']:
            raise ValueError('Cached token count differs')
        ids = row['output_ids']
        if any(t in STOPS for t in ids[:-1]):
            raise ValueError('Cached generation continued past an earlier EOS')
        result[cid] = row
    return result


def integrity_inventory(run_dir):
    run_dir=Path(run_dir)
    return {'manifest_sha256':file_hash(run_dir/'manifest.json'),
            'files':{('checkpoint-'+str(s)+'/outputs.jsonl'):file_hash(run_dir/('checkpoint-'+str(s))/'outputs.jsonl') for s in STEPS}}


def verify_frozen(run_dir):
    path=Path(run_dir)/'generation_integrity.json'
    if path.exists() and read_json(path)!=integrity_inventory(run_dir):
        raise ValueError('Completed dev raw outputs changed; preserve the frozen run')


def export(run_dir, *, final=False):
    run_dir = Path(run_dir);manifest = read_json(run_dir/'manifest.json');config = manifest['config']
    verify_frozen(run_dir)
    _,cases = cohort(config['prepared']);case_map = {c['case_id']:c for c in cases}
    observed = {str(s):cached_records(run_dir/('checkpoint-'+str(s))/'outputs.jsonl',case_map,
                                     step=s,signature=digest(manifest['preflight']['checkpoints'][str(s)])) for s in STEPS}
    common = set.intersection(*[set(v) for v in observed.values()])
    reports = run_dir/'reports';reports.mkdir(exist_ok=True)
    def rows_for(scope):
        result=[]
        for step,rows in observed.items():
            ids = sorted(common if scope=='common' else rows)
            values=[rows[cid] for cid in ids]
            r={'checkpoint':'checkpoint-'+step,'epoch':int(step)//501,'completed':len(rows),'planned':len(cases),'cohort_n':len(values),
               'scope':scope,'status':'complete' if len(rows)==len(cases) else 'partial'}
            for name in ['token_f1','rougeL','normalized_exact_match']:
                r[name] = sum(float(v['scores'][name]) for v in values)/len(values) if values else None
            r['token_budget_rate'] = sum(v['token_budget_reached'] for v in values)/len(values) if values else None
            r['mean_generated_tokens'] = sum(v['generated_tokens'] for v in values)/len(values) if values else None
            result.append(r)
        return result
    available,matched = rows_for('available'),rows_for('common')
    for filename,table in [('available.csv',available),('common.csv',matched)]:
        temporary = reports/(filename+'.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        temporary.replace(reports/filename)
    complete=all(len(v)==len(cases) for v in observed.values())
    write_json(reports/'summary.json',{'protocol':PROTOCOL,'complete':complete,'common_cases':len(common),'planned_per_checkpoint':len(cases),
                                      'available':available,'common':matched,'interpretation':'QA lexical utility on held-out ChatQA development rows; not RQ test results; no automatic winner.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    labels=['Epoch '+str(s//501) for s in STEPS]
    axes[0].bar(labels,[r['completed'] for r in available],color='#64748b');axes[0].set_ylim(0,len(cases)*1.15)
    axes[0].set_title('Committed outputs per checkpoint');axes[0].set_ylabel('Cases')
    for bar,row in zip(axes[0].patches,available):axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height(),str(row['completed'])+'/'+str(len(cases)),ha='center',va='bottom',fontsize=8)
    if common:
        axes[1].plot(labels,[r['token_f1'] for r in matched],marker='o',label='Token F1')
        axes[1].plot(labels,[r['rougeL'] for r in matched],marker='o',label='ROUGE-L F1');axes[1].legend()
    else:
        axes[1].text(.5,.5,'Pending common cases across all three epochs',ha='center',va='center',wrap=True,transform=axes[1].transAxes)
    axes[1].set_ylim(0,1);axes[1].set_title('Common cohort n='+str(len(common)));axes[1].set_ylabel('Lexical score [0,1]')
    fig.suptitle('ChatQA epoch dev diagnostic; '+('complete' if complete and final else 'provisional'))
    temp=reports/'progress.tmp.png';fig.savefig(temp,dpi=150);temp.replace(reports/'progress.png');plt.close(fig)
    return complete


def run(config, report):
    import torch
    from experiments.checkpoints import load_checkpoint
    from experiments.rq_live_io import append_durable
    from experiments.rq_week_runner import generation_lock,stop_signals
    if not report['ready'] or report['source_sha256'] != source_identity():
        raise ValueError('Resolve preflight errors/source changes first')
    manifest={'protocol':PROTOCOL,'config':config,'preflight':report}
    output=Path(config['output_root'])/digest(manifest);output.mkdir(parents=True,exist_ok=True)
    old=output/'manifest.json'
    if old.exists() and read_json(old)!=manifest:
        raise ValueError('Resume manifest differs')
    _,cases=cohort(config['prepared']);case_map={c['case_id']:c for c in cases}
    print('RUN_DIR='+str(output),flush=True)
    with generation_lock(output):
        verify_frozen(output)
        write_json(old,manifest)
        try:
            with stop_signals():
                for step in STEPS:
                    path=output/('checkpoint-'+str(step))/'outputs.jsonl'
                    signature=digest(report['checkpoints'][str(step)])
                    cached=cached_records(path,case_map,repair=True,step=step,signature=signature)
                    pending=[c for c in cases if c['case_id'] not in cached]
                    export(output)
                    if not pending:
                        continue
                    print('[LOAD] checkpoint-'+str(step),flush=True)
                    model,tokenizer,_=load_checkpoint(config['checkpoints'][str(step)],'ha_aside','qwen25_7b',torch.bfloat16,config['device'])
                    try:
                        with resolved_stops(model,STOPS):
                            for case in pending:
                                encoded=encode_case(tokenizer,case,MAX_CONTEXT)
                                if digest(encoded['rendered_prompt'])!=report['budgets'][case['case_id']]['transcript_sha256']:
                                    raise ValueError('Generation transcript differs from preflight')
                                inputs={k:encoded[k].to(model.device) for k in ['input_ids','segment_ids','binary_data_mask','attention_mask']}
                                start=time.perf_counter()
                                ids,_=model.generate(**inputs,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True)
                                continuation=ids[0,inputs['input_ids'].shape[1]:].tolist()
                                if any(t in STOPS for t in continuation[:-1]):
                                    raise ValueError('Generator continued past EOS')
                                text=tokenizer.decode(continuation,skip_special_tokens=True)
                                result={'case_id':case['case_id'],'case_sha256':digest(case),'source_group':case['row']['group_id'],
                                        'checkpoint_step':step,'checkpoint_signature':signature,'stop_ids':STOPS,'output':text,'output_ids':continuation,
                                        'generated_tokens':len(continuation),'token_budget_reached':len(continuation)==MAX_NEW,
                                        'seconds':time.perf_counter()-start,'scores':score_answer(text,case['row']['answers']),
                                        'question':case['row']['messages'][-1]['content'],'answers':case['row']['answers'],
                                        'rendered_prompt':encoded['rendered_prompt'],'role_spans':encoded['spans']}
                                result['record_sha256']=digest(result)
                                append_durable(path,result);cached[case['case_id']]=result
                                print('[DEV] checkpoint-'+str(step)+' '+str(len(cached))+'/'+str(len(cases))+' tokens='+str(len(continuation)),flush=True)
                                if len(cached)%10==0 or len(cached)==len(cases):
                                    export(output)
                    finally:
                        del model,tokenizer
                        gc.collect();torch.cuda.empty_cache()
                complete=export(output,final=True)
                if not complete:
                    raise ValueError('Missing source cases at completion')
                write_json(output/'generation_integrity.json',integrity_inventory(output))
                write_json(output/'status.json',{'status':'complete_development_diagnostic','steps':STEPS})
        except BaseException as error:
            write_json(output/'status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed','error':type(error).__name__+': '+str(error)})
            export(output)
            raise
    print(str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=PREPARED)
    p=sub.add_parser('setup');p.add_argument('--trial-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--prepared',type=Path,default=PREPARED);p.add_argument('--config',type=Path,required=True)
    for name in ['preflight','run']:
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True);p.add_argument('--report',type=Path)
    p=sub.add_parser('export');p.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.data_dir,args.output)
    elif args.command=='setup':setup(args)
    elif args.command=='export':export(args.run_dir)
    else:
        config=read_json(args.config);report=preflight(config)
        if args.report:write_json(args.report,report)
        print(json.dumps({k:v for k,v in report.items() if k in ['ready','errors','cases','dev_fingerprint']},indent=2),flush=True)
        if not report['ready']:raise SystemExit(2)
        if args.command=='run':run(config,report)


if __name__=='__main__':main()
