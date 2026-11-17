"""Resume-safe HA-only 16 -> 3 -> 1 search on frozen 10% / 30% / full ChatQA."""
from contextlib import redirect_stdout
import argparse
import csv
from functools import partial
import gc
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from experiments.common import digest,file_hash,write_json
from tools.train_matched_baselines import environment,read
from training.hpo_data import PROTOCOL

PREPARED=PROJECT/'data/prepared/ha_hpo_v1'
STAGES=['screen','refine','full']
STOPS=[151643,151645]
SOURCES=['tools/ha_hpo.py','training/hpo_data.py','training/hpo_runtime.py',
         'tools/prepare_chatqa_dataset.py','experiments/checkpoints.py',
         'tools/train_matched_baselines.py','training/resumable_chatqa.py','tools/check_ha_epoch_dev.py',
         'training/chatqa.py','training/dataset.py','experiments/common.py','experiments/prompts.py',
         'experiments/metrics.py','experiments/rq_transcript.py','experiments/rq_live_io.py',
         'ha_aside/hf_adapters.py','ha_aside/core.py','ha_aside/rgag.py']


def sealed(payload):
    return {**payload,'sha256':digest(payload)}


def verify_seal(payload):
    if payload.get('sha256')!=digest({k:v for k,v in payload.items() if k!='sha256'}):
        raise ValueError('Receipt/selection was edited or corrupted')
    return payload


def sources():return {name:file_hash(PROJECT/name) for name in SOURCES}


def grid():
    return [{'id':f'trial-{i:02d}','learning_rate':lr,'warmup_ratio':w,'effective_batch_size':batch,
             'per_device_train_batch_size':1,'gradient_accumulation_steps':batch}
            for i,(lr,w,batch) in enumerate(itertools.product([1e-6,5e-6,1e-5,2e-5],[0.,.1],[64,256]),1)]


def planned_steps(rows,trial,epochs=3):return math.ceil(rows/trial['effective_batch_size'])*epochs


def warmup_steps(rows,trial,epochs=3):return math.ceil(planned_steps(rows,trial,epochs)*trial['warmup_ratio'])


def folder_for(config,stage,trial_id):return Path(config['output_root'])/'trials'/stage/trial_id


def trial_spec(config,trial_id):
    return next(t for t in config['grid'] if t['id']==trial_id)


def setup(reference,data_dir,output,prepared=PREPARED,save_steps=25,keep_resume=2):
    from training.hpo_data import load
    from tools.train_matched_baselines import prepare as validate_reference
    frozen,manifest,splits,selected,cases=load(prepared,data_dir)
    # Reuse the audited reference check without creating any baseline run/setup.
    with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
        baseline=validate_reference(reference,data_dir,Path(tmp)/'reference-check',
                                    PROJECT.parent/'deployment/deepspeed_zero2_cpu_offload.json',save_steps,keep_resume)
    recipe=baseline['recipe']
    recipe.update(source_sha256=sources(),warmup_semantics='explicit integer ceil(warmup_ratio * planned_optimizer_steps)')
    config={'protocol':PROTOCOL,'output_root':str(Path(output).resolve()),'prepared':str(Path(prepared).resolve()),
            'data_dir':str(Path(data_dir).resolve()),'prepared_fingerprint':frozen['fingerprint'],
            'reference':baseline['reference'],'recipe':recipe,'grid':grid(),'epochs':3,
            'save_steps':save_steps,'keep_resume':keep_resume,'deepspeed':baseline['deepspeed'],'dev_cases':len(cases),
            'selection':{'screen_promote':3,'refine_promote':1,'primary':'token_f1','secondary':'rougeL',
                         'max_cap_without_EOS_rate':.05,'max_repeat_flag_rate':.05,
                         'ties':'F1 desc, ROUGE-L desc, cap asc, repetition asc, trial ID asc',
                         'if_no_eligible':'stop; never relax gates automatically'},
            'generation':{'max_new_tokens':1024,'max_context':16384,'stop_ids':STOPS,'do_sample':False},
            'retention':{'completed_subset_resume':'release only after all dev outputs and result receipt committed',
                         'subset_inference':'keep final epoch only','full_inference':'keep all three epochs',
                         'full_resume':'keep latest two full states; never prune for search automatically'},
            'source_sha256':sources()}
    config['fingerprint']=digest(config)
    output=Path(config['output_root']);output.mkdir(parents=True,exist_ok=True)
    path=output/'config.json'
    if path.exists():
        if read(path)!=config:raise FileExistsError('Existing HPO setup differs; choose a new root')
    elif any(output.iterdir()):raise FileExistsError('HPO output root is not empty')
    else:write_json(path,config)
    print('[SETUP] '+str(path));return config


def verify_config(config,full=True):
    if config['protocol']!=PROTOCOL or digest({k:v for k,v in config.items() if k!='fingerprint'})!=config['fingerprint']:
        raise ValueError('Frozen HPO configuration changed')
    if config['source_sha256']!=sources():raise ValueError('HPO source version changed; resume is locked')
    if file_hash(config['deepspeed'])!=config['recipe']['execution']['deepspeed']['sha256']:
        raise ValueError('DeepSpeed execution config changed')
    if read(Path(config['prepared'])/'manifest.json')['fingerprint']!=config['prepared_fingerprint']:
        raise ValueError('Prepared HPO selection changed')
    if full:
        from training.hpo_data import load
        data=load(config['prepared'],config['data_dir'])
        if data[0]['fingerprint']!=config['prepared_fingerprint']:raise ValueError('Prepared identity changed')
        return data


def preflight(config):
    report={'ready':False,'errors':[]}
    try:
        frozen,_,_,_,cases=verify_config(config)
        env=environment();report['environment']=env
        if not env['cuda_available']:report['errors'].append('CUDA unavailable in this Python')
        if env['packages']['deepspeed'] is None:report['errors'].append('DeepSpeed missing; use the original training venv, not Python base')
        else:
            import deepspeed
        for name,version in [('transformers','5.17.0'),('accelerate','1.15.0')]:
            if env['packages'][name]!=version:report['errors'].append(name+' must be '+version+' for this tested runner')
        import torch
        if env['cuda_available'] and not torch.cuda.is_bf16_supported():report['errors'].append('BF16 GPU required')
        if int(os.environ.get('WORLD_SIZE','1'))!=1:report['errors'].append('Single GPU/process only; run orchestrator without torchrun')
        report.update(subsets={s:{k:v for k,v in frozen['subsets'][s].items() if k not in ['groups','row_indices','strata']} for s in STAGES},
                      dev_cases=len(cases),trials={'screen':16,'refine':3,'full':1},
                      screen_steps={t['id']:planned_steps(frozen['subsets']['screen']['rows'],t,config['epochs']) for t in config['grid']},
                      source_strings_rewritten=False,disk_free_GB=round(shutil.disk_usage(config['output_root']).free/1e9,1),
                      scope='QA dev utility tuning; no RQ test optimization; not exhaustive full-data grid search')
    except Exception as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
    report['ready']=not report['errors'];return report


def runtime_identity(config):
    return {'config':config['fingerprint'],'environment':environment(),'python_executable':sys.executable}


def trial_identity(config,stage,trial_id):
    identity=runtime_identity(config)
    return digest({**identity,'stage':stage,'trial':trial_spec(config,trial_id)})


def native_inventory(path):
    path=Path(path)
    return {p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(path.iterdir())
            if p.is_file() and (p.suffix in ['.json','.safetensors','.jinja'] or p.name=='tokenizer.model')}


def training_receipt(config,stage,trial_id,hashes=False):
    folder=folder_for(config,stage,trial_id)
    path=folder/'training.json'
    if not path.exists():return None
    receipt=verify_seal(read(path))
    if receipt['identity']!=trial_identity(config,stage,trial_id):raise ValueError('Training receipt belongs to another runtime/trial')
    native=Path(receipt['checkpoint'])
    for name,item in receipt['files'].items():
        p=native/name
        if not p.is_file() or p.stat().st_size!=item['bytes'] or (hashes and file_hash(p)!=item['sha256']):
            raise ValueError('Completed inference checkpoint changed: '+str(p))
    return receipt


def train_worker(config,stage,trial_id,stop_after_step=None,require_cuda=True):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments
    from transformers import set_seed
    from training.dataset import HierarchicalConversationDataset,collate_hierarchical_batch
    from training.resumable_chatqa import run_lock,latest_snapshot
    from training.hpo_runtime import HPOTrainer,HPOStop
    from ha_aside.hf_adapters import HAASIDEModelWrapper
    frozen,parent,splits,subsets,cases=verify_config(config)
    folder=folder_for(config,stage,trial_id)
    trial=trial_spec(config,trial_id);identity=trial_identity(config,stage,trial_id)
    with run_lock(folder):
        if training_receipt(config,stage,trial_id):return 'complete'
        old_identity=folder/'identity.json'
        expected={'identity':identity,'stage':stage,'trial':trial,'runtime':runtime_identity(config)}
        if old_identity.exists() and read(old_identity)!=expected:raise ValueError('Training resume environment/recipe differs')
        write_json(old_identity,expected)
        latest,metadata,broken=latest_snapshot(folder,identity)
        for error in broken:
            p=Path(error['checkpoint']);p.resolve().relative_to((folder/'resume').resolve())
            quarantine=folder/'.quarantine';quarantine.mkdir(exist_ok=True)
            p.rename(quarantine/(p.name+'-'+uuid.uuid4().hex));print('[QUARANTINE]',error,flush=True)
        resumed_step=0 if metadata is None else metadata['step']
        for p in (folder/'epochs').glob('checkpoint-*'):
            if int(p.name.split('-')[-1])>resumed_step:
                quarantine=folder/'.quarantine';quarantine.mkdir(exist_ok=True)
                p.resolve().relative_to((folder/'epochs').resolve());p.rename(quarantine/(p.name+'-export-'+uuid.uuid4().hex))
        if latest is None and (folder/'status.json').exists() and read(folder/'status.json').get('step',0)>0:
            raise ValueError('Saved progress has no full resume state; refusing a silent restart')
        write_json(folder/'status.json',{'status':'training','step':resumed_step,'stage':stage,'trial':trial_id})
        rows=subsets[stage]
        validation=splits['validation'] if stage=='full' else [c['row'] for c in cases]
        recipe=config['recipe']
        total=planned_steps(len(rows),trial,config['epochs']);warm=warmup_steps(len(rows),trial,config['epochs'])
        stop=HPOStop(folder,Path(config['output_root']),stop_after_step,expected_steps=total)
        try:
            with stop.signal_handlers():
                set_seed(recipe['training_seed'])
                tokenizer=AutoTokenizer.from_pretrained(recipe['tokenizer_id'],revision=recipe['tokenizer_revision'],trust_remote_code=False)
                if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
                train=HierarchicalConversationDataset(rows,tokenizer)
                val=HierarchicalConversationDataset(validation,tokenizer)
                for ds in [train,val]:
                    for index in {0,len(ds)-1,max(range(len(ds)),key=lambda i:ds.data[i]['n_tokens'])}:ds[index]
                base=AutoModelForCausalLM.from_pretrained(recipe['base_model'],revision=recipe['base_revision'],
                     dtype=torch.bfloat16 if require_cuda else torch.float32,attn_implementation='eager',trust_remote_code=False)
                base.config.eos_token_id=tokenizer.eos_token_id;base.config.pad_token_id=tokenizer.pad_token_id;base.config.use_cache=False
                if recipe['gradient_checkpointing']:base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                model=HAASIDEModelWrapper(base,mode='ha_aside',enable_rgag=False,rotation_direction='right')
                provenance={**recipe,'trial':trial,'source_sha256':sources(),'epochs':config['epochs'],
                    'n_train':len(rows),'n_validation':len(validation),
                    'warmup_semantics':'explicit ceil(ratio * planned_optimizer_steps)',
                    'resolved_warmup_steps':warm,'planned_optimizer_steps':total,
                    'hpo':{'protocol':PROTOCOL,'config':config['fingerprint'],'prepared':frozen['fingerprint'],
                           'stage':stage,'trial_id':trial_id,'subset_row_identity':frozen['subsets'][stage]['row_identity_sha256'],
                           'full_grid_search_complete':False},
                    'execution':{**recipe['execution'],'torch':torch.__version__,'precision':'bfloat16' if require_cuda else 'float32',
                                 'runtime':environment()}}
                if stage!='full':
                    provenance.update(training_recipe='chatqa_hpo_subset_v1',
                         training_data_sha256=digest([r['source_row_sha256'] for r in rows]),
                         training_data_hash_scope='ordered unchanged source-row SHA list; full parent file hash remains in dataset_provenance',
                         dataset_provenance={'parent_manifest':parent,'subset':frozen['subsets'][stage]})
                args=TrainingArguments(output_dir=str(folder),num_train_epochs=config['epochs'],
                     learning_rate=trial['learning_rate'],lr_scheduler_type='cosine',warmup_steps=warm,
                     optim='adamw_torch',weight_decay=0.,max_grad_norm=1.,
                     per_device_train_batch_size=trial['per_device_train_batch_size'],per_device_eval_batch_size=1,
                     gradient_accumulation_steps=trial['gradient_accumulation_steps'],
                     eval_strategy='epoch',save_strategy='steps',save_steps=config['save_steps'],save_total_limit=None,
                     save_only_model=False,logging_steps=10,bf16=require_cuda,report_to='none',
                     remove_unused_columns=False,label_names=['labels'],seed=recipe['training_seed'],data_seed=recipe['training_seed'],
                     deepspeed=config['deepspeed'] if require_cuda else None,prediction_loss_only=True,dataloader_num_workers=0,ignore_data_skip=False)
                if args.get_warmup_steps(total)!=warm:raise ValueError('Warmup semantic drift')
                trainer=HPOTrainer(model=model,args=args,train_dataset=train,eval_dataset=val,
                    data_collator=partial(collate_hierarchical_batch,pad_token_id=tokenizer.pad_token_id),callbacks=[stop],
                    run_identity=identity,provenance=provenance,export_tokenizer=tokenizer,keep_resume=config['keep_resume'],hpo_stage=stage)
                print('[RESUME]' if latest else '[TRAIN]',stage,trial_id,'step='+str(resumed_step),'total='+str(total),'warmup_steps='+str(warm),flush=True)
                if metadata is not None and metadata['step']==total:
                    # A crash after final commit but before the small training receipt.
                    # The native export was committed before this final resume snapshot.
                    state=read(latest/'trainer_state.json')
                else:
                    trainer.train(resume_from_checkpoint=str(latest) if latest else None)
                    state={'global_step':trainer.state.global_step,'max_steps':trainer.state.max_steps,'epoch':trainer.state.epoch}
                step=state['global_step']
                if step<total:
                    write_json(folder/'status.json',{'status':'stopped','step':step});return 'stopped'
                chosen=read(folder/'selected_checkpoint.json')['checkpoint']
                receipt=sealed({'identity':identity,'stage':stage,'trial_id':trial_id,'step':step,'max_steps':total,
                               'warmup_steps':warm,'checkpoint':chosen,'files':native_inventory(chosen)})
                write_json(folder/'training.json',receipt)
                write_json(folder/'status.json',{'status':'trained','step':step,'checkpoint':chosen})
                return 'complete'
        except BaseException as exc:
            write_json(folder/'status.json',{'status':'interrupted_or_failed','step':resumed_step,'error':str(exc)});raise


def repeat_flag(text):
    words=re.findall(r'\w+',text.casefold())
    from collections import Counter
    counts=Counter(tuple(words[i:i+4]) for i in range(len(words)-3))
    return max(counts.values(),default=0)>=5


def output_records(config,stage,trial_id,cases,repair=False):
    from experiments.rq_live_io import committed_rows
    folder=folder_for(config,stage,trial_id);receipt=training_receipt(config,stage,trial_id)
    if receipt is None:raise ValueError('Trial is not fully trained')
    expected={c['case_id']:c for c in cases};observed={}
    rows,_=committed_rows(folder/'dev_outputs.jsonl',repair=repair)
    for row in rows:
        verify_seal(row)
        cid=row['case_id']
        if cid in observed or cid not in expected or row['case_sha256']!=digest(expected[cid]) or row['training_receipt']!=receipt['sha256']:
            raise ValueError('Dev cache has changed source/gold/checkpoint or duplicate ID')
        if row['stop_ids']!=config['generation']['stop_ids'] or any(i in row['stop_ids'] for i in row['output_ids'][:-1]):
            raise ValueError('Dev cache has incompatible EOS policy')
        observed[cid]=row
    return observed


def summarize(records,planned):
    rows=list(records.values());n=len(rows)
    return {'completed':n,'planned':planned,'complete':n==planned,
            **{key:sum(r['scores'][key] for r in rows)/n if n else None for key in ['token_f1','rougeL','normalized_exact_match']},
            'cap_without_EOS_rate':sum(r['cap_without_EOS'] for r in rows)/n if n else None,
            'repeat_flag_rate':sum(r['repeat_flag'] for r in rows)/n if n else None}


def score_worker(config,stage,trial_id,require_cuda=True,stop_after_cases=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from ha_aside.hf_adapters import HAASIDEModelWrapper
    from tools.check_ha_epoch_dev import encode_case,score_answer,resolved_stops
    from training.resumable_chatqa import run_lock,StopAtBoundary
    from experiments.rq_live_io import append_durable
    _,_,_,_,cases=verify_config(config)
    folder=folder_for(config,stage,trial_id)
    with run_lock(folder):
        receipt=training_receipt(config,stage,trial_id,hashes=True)
        if receipt is None:raise ValueError('Scoring requires a completed training receipt')
        observed=output_records(config,stage,trial_id,cases,repair=True)
        guard=StopAtBoundary(folder);root=Path(config['output_root'])
        if len(observed)<len(cases):
            tokenizer=AutoTokenizer.from_pretrained(receipt['checkpoint'],local_files_only=True,trust_remote_code=False)
            native=read(Path(receipt['checkpoint'])/'architecture.json')
            hpo=native['provenance']['hpo']
            if (native['method']!='ha_aside' or native['enable_rgag'] or hpo['stage']!=stage or hpo['trial_id']!=trial_id
                    or hpo['config']!=config['fingerprint'] or hpo['prepared']!=config['prepared_fingerprint']
                    or native['rotation_direction']!='right'
                    or {str(k):v for k,v in native['angles'].items()}!={'0':0.,'1':math.pi/8,'2':math.pi/2,'3':0.}):
                raise ValueError('Inference method/role encoding does not match the trial')
            base=AutoModelForCausalLM.from_pretrained(receipt['checkpoint'],dtype=torch.bfloat16 if require_cuda else torch.float32,
                 attn_implementation='eager',trust_remote_code=False,local_files_only=True)
            model=HAASIDEModelWrapper(base,mode='ha_aside',enable_rgag=False,
                  angles={int(k):v for k,v in native['angles'].items()},rotation_direction=native['rotation_direction'])
            model.to('cuda:0' if require_cuda else 'cpu').eval()
            with guard.signal_handlers(),resolved_stops(model,config['generation']['stop_ids']):
                for case in cases:
                    if case['case_id'] in observed:continue
                    if guard.requested_signal or (root/'STOP_REQUESTED').exists():break
                    encoded=encode_case(tokenizer,case,config['generation']['max_context'])
                    inputs={k:encoded[k].to(model.device) for k in ['input_ids','segment_ids','binary_data_mask','attention_mask']}
                    with torch.inference_mode():
                        ids,_=model.generate(**inputs,max_new_tokens=config['generation']['max_new_tokens'],do_sample=False,use_cache=True)
                    continuation=ids[0,inputs['input_ids'].shape[1]:].tolist()
                    if any(i in config['generation']['stop_ids'] for i in continuation[:-1]):raise ValueError('Generation continued past EOS')
                    text=tokenizer.decode(continuation,skip_special_tokens=True)
                    record=sealed({'case_id':case['case_id'],'case_sha256':digest(case),'training_receipt':receipt['sha256'],
                         'stop_ids':config['generation']['stop_ids'],'output':text,'output_ids':continuation,
                         'scores':score_answer(text,case['row']['answers']),
                         'cap_without_EOS':len(continuation)==config['generation']['max_new_tokens'] and continuation[-1] not in config['generation']['stop_ids'],
                         'repeat_flag':repeat_flag(text),'rendered_prompt_sha256':digest(encoded['rendered_prompt'])})
                    append_durable(folder/'dev_outputs.jsonl',record);observed[case['case_id']]=record
                    if len(observed)%25==0:print('[DEV]',stage,trial_id,str(len(observed))+'/'+str(len(cases)),flush=True)
                    write_json(folder/'dev_progress.json',summarize(observed,len(cases)))
                    if stop_after_cases is not None and len(observed)>=stop_after_cases:break
            del model,base;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        summary=summarize(observed,len(cases))
        if summary['complete']:
            result=sealed({'protocol':PROTOCOL,'config':config['fingerprint'],'stage':stage,'trial_id':trial_id,
                 'training_receipt':receipt['sha256'],'dev_outputs_sha256':file_hash(folder/'dev_outputs.jsonl'),**summary})
            path=folder/'result.json'
            if path.exists() and read(path)!=result:raise ValueError('Completed dev result changed')
            write_json(path,result);print('[SCORED]',stage,trial_id,summary,flush=True)
        return 'complete' if summary['complete'] else 'stopped'


def result_for(config,stage,trial_id):
    folder=folder_for(config,stage,trial_id);path=folder/'result.json'
    if not path.exists():return None
    result=verify_seal(read(path))
    if (result['config']!=config['fingerprint'] or result['stage']!=stage or result['trial_id']!=trial_id
            or not result['complete'] or result['completed']!=result['planned'] or result['planned']!=config['dev_cases']
            or result['dev_outputs_sha256']!=file_hash(folder/'dev_outputs.jsonl')):
        raise ValueError('Completed result is incomplete or changed')
    receipt=training_receipt(config,stage,trial_id)
    if receipt is None or result['training_receipt']!=receipt['sha256']:raise ValueError('Result checkpoint binding changed')
    return result


def choose(config,stage,ids,count):
    results=[]
    for trial_id in ids:
        result=result_for(config,stage,trial_id)
        if result is None:raise ValueError('All trials must finish dev scoring before promotion: '+trial_id)
        results.append(result)
    policy=config['selection']
    eligible=[r for r in results if r['cap_without_EOS_rate']<=policy['max_cap_without_EOS_rate'] and r['repeat_flag_rate']<=policy['max_repeat_flag_rate']]
    eligible.sort(key=lambda r:(-r['token_f1'],-r['rougeL'],r['cap_without_EOS_rate'],r['repeat_flag_rate'],r['trial_id']))
    if not eligible:raise ValueError('No trial passes frozen quality gates; search stops without choosing a winner')
    choice=sealed({'config':config['fingerprint'],'stage':stage,'result_sha256':{r['trial_id']:r['sha256'] for r in results},
                   'ranked_eligible':[r['trial_id'] for r in eligible],'promoted':[r['trial_id'] for r in eligible[:count]],
                   'requested_promotions':count,'policy':policy})
    path=Path(config['output_root'])/(stage+'_selection.json')
    if path.exists() and read(path)!=choice:raise ValueError('Frozen promotion changed')
    write_json(path,choice);return choice['promoted']


def cleanup_scored(config,stage,trial_id):
    """Keep final subset weights/results, release only verified completed state."""
    if stage=='full':return
    result=result_for(config,stage,trial_id)
    if result is None:raise ValueError('Never release an unscored trial')
    from training.resumable_chatqa import validate_snapshot
    folder=folder_for(config,stage,trial_id);receipt=training_receipt(config,stage,trial_id)
    commits=[]
    for p in (folder/'resume').glob('checkpoint-*'):
        commits.append(validate_snapshot(p,receipt['identity'],hashes=False))
    if commits:
        write_json(folder/'released_resume_inventory.json',sealed({'result':result['sha256'],'commits':commits}))
        for p in (folder/'resume').glob('checkpoint-*'):
            p.resolve().relative_to((folder/'resume').resolve());shutil.rmtree(p)
    pending=folder/'.pending'
    if pending.exists():
        pending.resolve().relative_to(folder.resolve());shutil.rmtree(pending)


def report(config,write=True):
    root=Path(config['output_root']);rows=[]
    frozen=read(Path(config['prepared'])/'manifest.json')
    for stage in STAGES:
        for folder in sorted((root/'trials'/stage).glob('trial-*')):
            trial=folder.name;value=result_for(config,stage,trial)
            state=read(folder/'status.json') if (folder/'status.json').exists() else {'status':'not_started'}
            latest=read(folder/'latest_checkpoint.json') if (folder/'latest_checkpoint.json').exists() else {}
            progress={'optimizer_step':latest.get('step',state.get('step',0)),
                      'planned_optimizer_steps':planned_steps(frozen['subsets'][stage]['rows'],trial_spec(config,trial),config['epochs'])}
            if value is not None:rows.append({'stage':stage,'trial_id':trial,'status':'scored',**progress,**{k:value[k] for k in ['completed','planned','token_f1','rougeL','cap_without_EOS_rate','repeat_flag_rate']}})
            elif (folder/'dev_progress.json').exists():
                value=read(folder/'dev_progress.json');rows.append({'stage':stage,'trial_id':trial,'status':'partial dev',**progress,**{k:value[k] for k in ['completed','planned','token_f1','rougeL','cap_without_EOS_rate','repeat_flag_rate']}})
            else:
                rows.append({'stage':stage,'trial_id':trial,'status':state['status'],**progress,'completed':0,'planned':config['dev_cases'],'token_f1':None,'rougeL':None,'cap_without_EOS_rate':None,'repeat_flag_rate':None})
    if not write:return rows
    dest=root/'reports';dest.mkdir(exist_ok=True)
    nonce=uuid.uuid4().hex
    temporary=dest/(nonce+'-progress.json')
    write_json(temporary,{'rows':rows,'scope':'Partial trials are never eligible for promotion; stage cohorts differ'})
    temporary.replace(dest/'progress.json')
    if rows:
        temporary=dest/(nonce+'-progress.csv')
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary.replace(dest/'progress.csv')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
        for ax,stage in zip(axes,STAGES):
            values=[r for r in rows if r['stage']==stage and r['status']=='scored']
            if values:
                ax.bar([r['trial_id'].replace('trial-','') for r in values],[r['token_f1']*100 for r in values],color='#2454ed')
            ax.set(title=stage+' (completed dev only)',ylabel='Token F1 (0-100)',ylim=(0,100));ax.tick_params(axis='x',labelrotation=60)
        temporary=dest/(nonce+'-progress.png')
        fig.savefig(temporary,dpi=170);plt.close(fig);temporary.replace(dest/'progress.png')
    return rows


def run_search(config,through='full',stop_after_step=None):
    from training.resumable_chatqa import run_lock
    verify_config(config,full=False)
    root=Path(config['output_root'])
    with run_lock(root):
        readiness=preflight(config);write_json(root/'preflight.json',readiness)
        if not readiness['ready']:raise ValueError('Resolve preflight errors: '+str(readiness['errors']))
        runtime=runtime_identity(config)
        if (root/'runtime.json').exists() and read(root/'runtime.json')!=runtime:raise ValueError('Runtime changed since this search started')
        write_json(root/'runtime.json',runtime)
        if (root/'STOP_REQUESTED').exists():(root/'STOP_REQUESTED').unlink()
        ids=[t['id'] for t in config['grid']]
        for stage in STAGES[:STAGES.index(through)+1]:
            for trial_id in ids:
                if (root/'STOP_REQUESTED').exists():
                    write_json(root/'status.json',{'status':'stopped','stage':stage,'trial_id':trial_id});report(config);return
                if result_for(config,stage,trial_id):
                    cleanup_scored(config,stage,trial_id);continue
                write_json(root/'status.json',{'status':'running','stage':stage,'trial_id':trial_id})
                folder=folder_for(config,stage,trial_id);folder.mkdir(parents=True,exist_ok=True)
                for worker in ['worker-train','worker-score']:
                    if worker=='worker-train' and training_receipt(config,stage,trial_id):continue
                    if (root/'STOP_REQUESTED').exists():break
                    base=[sys.executable,'-u']
                    if worker=='worker-train':base+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1','--']
                    command=base+[str(Path(__file__).resolve()),worker,'--config',str(root/'config.json'),'--stage',stage,'--trial',trial_id]
                    if worker=='worker-train' and stop_after_step is not None:command+=['--stop-after-step',str(stop_after_step)]
                    print('[RUN]',stage,trial_id,worker,flush=True)
                    with (folder/(worker+'.log')).open('a',encoding='utf-8') as logfile:
                        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
                        try:
                            for line in proc.stdout:
                                logfile.write(line);logfile.flush();print(line,end='',flush=True)
                            code=proc.wait()
                        except KeyboardInterrupt:
                            (root/'STOP_REQUESTED').write_bytes(b'cooperative stop\n')
                            # Workers observe the request at safe boundaries; keep
                            # draining stdout so a full pipe cannot block saving.
                            for line in proc.stdout:logfile.write(line);logfile.flush();print(line,end='',flush=True)
                            code=proc.wait()
                    if code:
                        state='stopped' if (root/'STOP_REQUESTED').exists() else 'failed'
                        write_json(root/'status.json',{'status':state,'stage':stage,'trial_id':trial_id,'worker_exit_code':code})
                        report(config)
                        if state=='stopped':print('[STOPPED] Worker interrupted; resume last committed checkpoint.',flush=True);return
                        raise RuntimeError(worker+' failed; preserved log '+str(folder/(worker+'.log')))
                    if worker=='worker-train' and training_receipt(config,stage,trial_id) is None:break
                if result_for(config,stage,trial_id) is None:
                    write_json(root/'status.json',{'status':'stopped','stage':stage,'trial_id':trial_id});report(config)
                    print('[STOPPED] Resume with the same run command.',flush=True);return
                cleanup_scored(config,stage,trial_id);report(config)
            if stage in ['screen','refine']:
                try:
                    ids=choose(config,stage,ids,config['selection'][stage+'_promote'])
                except ValueError as exc:
                    write_json(root/'status.json',{'status':'selection_blocked','stage':stage,'error':str(exc)})
                    report(config);raise
            else:
                result=result_for(config,stage,ids[0]);receipt=training_receipt(config,stage,ids[0])
                write_json(root/'selected_checkpoint.json',sealed({'checkpoint':receipt['checkpoint'],'trial':trial_spec(config,ids[0]),
                     'selection':'best dev-eligible refine candidate, then fresh full-data three-epoch refit',
                     'protocol':PROTOCOL,'full_grid_search_complete':False,'full_dev_result':result['sha256']}))
            write_json(root/'status.json',{'status':'complete' if stage=='full' else 'stage_complete','stage':stage,'next_trials':ids})
        print('[COMPLETE]' if through=='full' else '[STAGE COMPLETE]',through,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    data=sub.add_parser('prepare-data');data.add_argument('--data-dir',type=Path,required=True)
    data.add_argument('--output',type=Path,default=PREPARED);data.add_argument('--rq-dir',type=Path,default=PROJECT/'data/prepared/rq_public_v1')
    create=sub.add_parser('setup');create.add_argument('--ha-reference',type=Path,required=True);create.add_argument('--data-dir',type=Path,required=True)
    create.add_argument('--output-root',type=Path,required=True);create.add_argument('--prepared',type=Path,default=PREPARED)
    create.add_argument('--save-steps',type=int,default=25);create.add_argument('--keep-resume',type=int,default=2)
    for name in ['preflight','run','stop','status','report','worker-train','worker-score']:
        parser=sub.add_parser(name);parser.add_argument('--config',type=Path,required=True)
        if name=='run':parser.add_argument('--through',choices=STAGES,default='full')
        if name in ['run','worker-train']:parser.add_argument('--stop-after-step',type=int)
        if name in ['worker-train','worker-score']:
            parser.add_argument('--stage',choices=STAGES,required=True);parser.add_argument('--trial',required=True)
    args=p.parse_args()
    if getattr(args,'stop_after_step',None) is not None and args.stop_after_step<1:p.error('stop-after-step must be positive')
    if args.command=='prepare-data':
        from training.hpo_data import prepare
        frozen=prepare(args.data_dir,args.output,args.rq_dir)
        print(json.dumps({'fingerprint':frozen['fingerprint'],'audit':frozen['audit'],
                         'subsets':{s:{k:v for k,v in r.items() if k not in ['row_indices','groups','strata']} for s,r in frozen['subsets'].items()}},indent=2));return
    if args.command=='setup':setup(args.ha_reference,args.data_dir,args.output_root,args.prepared,args.save_steps,args.keep_resume);return
    config=read(args.config);root=Path(config['output_root'])
    if args.command=='preflight':
        value=preflight(config);write_json(root/'preflight.json',value);print(json.dumps(value,indent=2));sys.exit(0 if value['ready'] else 1)
    elif args.command=='run':run_search(config,args.through,args.stop_after_step)
    elif args.command=='stop':
        verify_config(config,full=False);(root/'STOP_REQUESTED').write_bytes(b'cooperative pipeline stop\n')
        print('[STOP REQUESTED] Wait for [STOPPED]; do not kill a worker while checkpointing.')
    elif args.command in ['report','status']:
        print(json.dumps({'pipeline':read(root/'status.json') if (root/'status.json').exists() else 'not_started','progress':report(config,write=args.command=='report')},indent=2))
    elif args.command=='worker-train':train_worker(config,args.stage,args.trial,args.stop_after_step)
    else:score_worker(config,args.stage,args.trial)


if __name__=='__main__':main()
