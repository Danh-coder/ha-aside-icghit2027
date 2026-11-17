"""Prepare/run/resume fixed-recipe ASIDE and Vanilla; no grid and no legacy edits."""
import argparse
from functools import partial
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.common import REGISTRY, digest, file_hash, write_json

PROTOCOL = 'chatqa_matched_baselines_resumable_v1'
METHODS = ['aside_orig', 'vanilla']
SOURCES = ['tools/train_matched_baselines.py', 'training/resumable_chatqa.py',
           'training/dataset.py', 'training/chatqa.py', 'experiments/prompts.py',
           'experiments/common.py', 'ha_aside/hf_adapters.py', 'ha_aside/core.py', 'ha_aside/rgag.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    return {name: file_hash(PROJECT / name) for name in SOURCES}


def environment():
    import torch
    packages = {}
    for name in ['transformers', 'accelerate', 'deepspeed', 'safetensors', 'tokenizers', 'numpy']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'packages': packages}


def prepare(reference, data_dir, output, deepspeed, save_steps=25, keep_resume=2):
    from tools.prepare_chatqa_dataset import load_prepared
    from training.chatqa import RECIPE, ROLE_POLICY, TOKENIZER_REVISION
    from experiments.checkpoints import checkpoint_metadata
    reference, data_dir, output = [Path(p).resolve() for p in [reference, data_dir, output]]
    metadata = checkpoint_metadata(reference, 'ha_aside', 'qwen25_7b')
    origin = metadata['provenance']
    state = read(reference / 'trainer_state.json')
    expected = {'learning_rate': 1e-6, 'warmup_ratio': 0., 'effective_batch_size': 64,
                'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 64}
    if (metadata['enable_rgag'] or metadata['rotation_direction'] != 'right'
            or {str(k): v for k,v in metadata['angles'].items()} != {'0':0., '1':math.pi/8, '2':math.pi/2, '3':0.}
            or origin.get('pilot_max_steps') is not None or origin['trial'] != expected
            or origin['epochs'] != 3 or state['global_step'] != 1503 or state['max_steps'] != 1503
            or abs(state['epoch']-3) > 1e-7 or origin['tokenizer_revision'] != TOKENIZER_REVISION):
        raise ValueError('Requires the completed RGAG-OFF 1503-step HA reference trial, not a grid winner or pilot')
    execution = origin['execution']
    if (execution['world_size'] != 1 or execution['optimizer'] != 'adamw_torch'
            or execution['weight_decay'] != 0. or execution['max_grad_norm'] != 1.
            or not origin.get('gradient_checkpointing')):
        raise ValueError('Unexpected HA execution recipe')
    evaluation=[r for r in state['log_history'] if 'eval_loss' in r and r['step']==1503]
    if not evaluation or abs(evaluation[-1]['eval_loss']-origin['selected_validation_loss'])>1e-9:
        raise ValueError('Missing matching final-epoch validation evidence')
    deepspeed = Path(deepspeed).resolve()
    if not execution['deepspeed'] or file_hash(deepspeed) != execution['deepspeed']['sha256']:
        raise ValueError('DeepSpeed CPU-offload config must match the actual HA run')
    manifest, splits = load_prepared(data_dir)
    if (digest(manifest) != origin['dataset_manifest_canonical_sha256']
            or file_hash(data_dir/'train.jsonl') != origin['training_data_sha256']
            or file_hash(data_dir/'validation.jsonl') != origin['validation_data_sha256']
            or [len(splits[k]) for k in ['train','validation']] != [32046,3552]):
        raise ValueError('Training corpus differs from the completed HA reference')
    if save_steps < 1 or keep_resume < 1:
        raise ValueError('save_steps and keep_resume must be positive')
    recipe = {k: origin[k] for k in ['training_recipe','dataset','base_model','tokenizer_id',
              'base_revision','tokenizer_revision','training_seed','epochs','loss_scope','max_length',
              'training_truncation','training_data_sha256','validation_data_sha256','n_train','n_validation',
              'role_policy','dataset_manifest_canonical_sha256','trial','gradient_checkpointing']}
    recipe.update(dataset_manifest_sha256=file_hash(data_dir/'manifest.json'), dataset_provenance=manifest,
                  enable_rgag=False, pilot_max_steps=None,
                  reference_training_execution=execution, source_sha256=sources(),
                  checkpoint_policy='full optimizer/scheduler/RNG resume snapshots; separate all-epoch inference exports',
                  warmup_semantics='legacy_grid_warmup_ratio_value_passed_as_warmup_steps',
                  execution={**execution,'deepspeed':{'path':str(deepspeed),'sha256':file_hash(deepspeed)}})
    config = {'protocol':PROTOCOL, 'reference':{'checkpoint':str(reference),
              'architecture_sha256':file_hash(reference/'architecture.json'),
              'trainer_state_sha256':file_hash(reference/'trainer_state.json'),
              'selection':'one completed HA trial, fixed three-epoch reference; not full-search selection'},
              'data_dir':str(data_dir), 'output_root':str(output), 'methods':METHODS,
              'recipe':recipe,'deepspeed':str(deepspeed), 'save_steps':save_steps,
              'keep_resume':keep_resume, 'source_sha256':sources()}
    config['fingerprint'] = digest(config)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'config.json'
    if path.exists():
        if read(path) != config:
            raise FileExistsError('Existing setup differs; select a new output root')
    elif any(output.iterdir()):
        raise FileExistsError('Output root is not empty')
    else:
        write_json(path,config)
    print('[SETUP] '+str(path))
    return config


def verify_config(config):
    if config['protocol'] != PROTOCOL or digest({k:v for k,v in config.items() if k!='fingerprint'}) != config['fingerprint']:
        raise ValueError('Training config was edited after setup')
    if config['source_sha256'] != sources():
        raise ValueError('Training implementation changed; do not mix source versions on resume')
    if file_hash(config['deepspeed']) != config['recipe']['execution']['deepspeed']['sha256']:
        raise ValueError('DeepSpeed config changed')
    from tools.prepare_chatqa_dataset import load_prepared
    manifest, splits = load_prepared(config['data_dir'])
    if digest(manifest) != config['recipe']['dataset_manifest_canonical_sha256']:
        raise ValueError('Data manifest changed')
    return splits


def preflight(config, require_cuda=True):
    verify_config(config)
    env = environment()
    errors = []
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        errors.append('Use exactly one process/GPU to match HA')
    if require_cuda and not env['cuda_available']:
        errors.append('CUDA is unavailable in this Python environment')
    if require_cuda and env['cuda_available']:
        import torch
        if not torch.cuda.is_bf16_supported():errors.append('GPU must support the HA bfloat16 recipe')
    if require_cuda and env['packages']['deepspeed'] is None:
        errors.append('The existing training environment needs DeepSpeed')
    if env['packages']['transformers'] != '5.17.0':
        errors.append('This resume runner was tested against Transformers 5.17.0; use that existing environment')
    if env['packages']['accelerate'] != '1.15.0':
        errors.append('This resume runner was tested against Accelerate 1.15.0')
    import shutil
    return {'ready':not errors, 'errors':errors, 'environment':env,
            'methods':config['methods'], 'train_rows':config['recipe']['n_train'],
            'validation_rows':config['recipe']['n_validation'], 'optimizer_steps_per_method':1503,
            'save_every_optimizer_steps':config['save_steps'], 'keep_full_resume_snapshots':config['keep_resume'],
            'disk_free_GB':round(shutil.disk_usage(config['output_root']).free/1e9,1),
            'resume_storage_note':'Full 7B AdamW/DeepSpeed snapshots can exceed 100 GB each; pending save temporarily adds one more.'}


def run(config, method, stop_after_step=None, *, require_cuda=True):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
    from training.dataset import HierarchicalConversationDataset, collate_hierarchical_batch
    from ha_aside.hf_adapters import HAASIDEModelWrapper
    from training.resumable_chatqa import ResumableTrainer, StopAtBoundary, run_lock, latest_snapshot
    if method not in config['methods']:
        raise ValueError('Method not configured')
    folder = Path(config['output_root'])/method
    with run_lock(Path(config['output_root'])), run_lock(folder):
        report = preflight(config, require_cuda=require_cuda)
        if not report['ready']:
            raise ValueError('Preflight is not ready: '+str(report['errors']))
        identity = {'config_fingerprint':config['fingerprint'], 'method':method,'environment':report['environment']}
        fingerprint = digest(identity)
        run_manifest = folder/'run_manifest.json'
        if run_manifest.exists():
            if read(run_manifest) != identity:
                raise ValueError('Resume environment/method/recipe changed')
        else:
            write_json(run_manifest,identity)
        latest, metadata, failures = latest_snapshot(folder,fingerprint)
        for failure in failures:
            print('[INVALID CHECKPOINT] '+str(failure),flush=True)
            broken = Path(failure['checkpoint'])
            broken.resolve().relative_to((folder/'resume').resolve())
            quarantine=folder/'.quarantine';quarantine.mkdir(exist_ok=True)
            broken.rename(quarantine/(broken.name+'-'+uuid.uuid4().hex))
        if latest is not None and metadata['step'] >= metadata['max_steps']:
            if not (folder/'selected_checkpoint.json').is_file():
                raise ValueError('Final checkpoint missing inference export; inspect the preserved run')
            write_json(folder/'status.json',{'status':'complete','step':metadata['step'],'checkpoint':str(latest)})
            print('[COMPLETE] '+str(folder/'selected_checkpoint.json'),flush=True)
            return
        if latest is None and (folder/'status.json').exists():
            previous = read(folder/'status.json')
            if previous.get('step',0)>0 or previous['status']=='complete':
                raise ValueError('Prior progress exists but no resume checkpoint; refusing to retrain from zero')
        # An interrupted save may have published an epoch export just before its
        # resume snapshot committed. Preserve it separately before replaying.
        resumed_step=0 if metadata is None else metadata['step']
        future_exports=[p for p in (folder/'epochs').glob('checkpoint-*') if int(p.name.split('-')[-1])>resumed_step]
        for exported in future_exports:
            exported.resolve().relative_to((folder/'epochs').resolve())
            quarantine=folder/'.quarantine';quarantine.mkdir(exist_ok=True)
            exported.rename(quarantine/(exported.name+'-export-'+uuid.uuid4().hex))
        if future_exports and (folder/'selected_checkpoint.json').exists():
            (folder/'selected_checkpoint.json').rename(folder/'.quarantine'/('selected-'+uuid.uuid4().hex+'.json'))
        if (folder/'STOP_REQUESTED').exists():
            (folder/'STOP_REQUESTED').unlink()
        status = {'status':'running','method':method,'step':0 if metadata is None else metadata['step'],
                  'resumed_from':None if latest is None else str(latest)}
        write_json(folder/'status.json',status)
        try:
            stop = StopAtBoundary(folder,stop_after_step)
            with stop.signal_handlers():
                recipe = config['recipe']
                splits = verify_config(config)
                set_seed(recipe['training_seed'])
                tokenizer = AutoTokenizer.from_pretrained(recipe['tokenizer_id'],revision=recipe['tokenizer_revision'],trust_remote_code=False)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                datasets = {key:HierarchicalConversationDataset(value,tokenizer) for key,value in splits.items()}
                for dataset in datasets.values():
                    for index in {0,len(dataset)-1,max(range(len(dataset)),key=lambda i:dataset.data[i]['n_tokens'])}:
                        dataset[index]
                base = AutoModelForCausalLM.from_pretrained(recipe['base_model'],revision=recipe['base_revision'],
                       dtype=torch.bfloat16 if require_cuda else torch.float32,attn_implementation='eager',trust_remote_code=False)
                base.config.eos_token_id=tokenizer.eos_token_id
                base.config.pad_token_id=tokenizer.pad_token_id
                base.config.use_cache=False
                if recipe['gradient_checkpointing']:
                    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                model=HAASIDEModelWrapper(base,mode=method,enable_rgag=False,rotation_direction='right')
                trial=recipe['trial']
                args=TrainingArguments(output_dir=str(folder),num_train_epochs=recipe['epochs'],
                     learning_rate=trial['learning_rate'],lr_scheduler_type='cosine',warmup_steps=trial['warmup_ratio'],
                     optim='adamw_torch',weight_decay=0.,max_grad_norm=1.,
                     per_device_train_batch_size=trial['per_device_train_batch_size'],per_device_eval_batch_size=trial['per_device_train_batch_size'],
                     gradient_accumulation_steps=trial['gradient_accumulation_steps'],
                     eval_strategy='epoch',save_strategy='steps',save_steps=config['save_steps'],save_total_limit=None,
                     save_only_model=False,logging_steps=10,bf16=require_cuda,report_to='none',
                     remove_unused_columns=False,label_names=['labels'],seed=recipe['training_seed'],data_seed=recipe['training_seed'],
                     deepspeed=config['deepspeed'] if require_cuda else None,prediction_loss_only=True,
                     dataloader_num_workers=0,ignore_data_skip=False)
                trainer=ResumableTrainer(model=model,args=args,train_dataset=datasets['train'],eval_dataset=datasets['validation'],
                      data_collator=partial(collate_hierarchical_batch,pad_token_id=tokenizer.pad_token_id),
                      callbacks=[stop],run_identity=fingerprint,
                      provenance={**recipe,'execution':{**recipe['execution'],'torch':torch.__version__,
                                  'precision':'bfloat16' if require_cuda else 'float32',
                                  'transformers':report['environment']['packages']['transformers'],
                                  'accelerate':report['environment']['packages']['accelerate']}},export_tokenizer=tokenizer,
                      keep_resume=config['keep_resume'])
                print('[START]' if latest is None else '[RESUME]',method,'step='+str(status['step']),
                      'schedule remains '+str(recipe['epochs'])+' epochs',flush=True)
                trainer.train(resume_from_checkpoint=None if latest is None else str(latest))
                step=trainer.state.global_step
                complete=step>=trainer.state.max_steps
                # The committed snapshot is the reliable source of resumable progress.
                snapshot,snapshot_meta,_=latest_snapshot(folder,fingerprint)
                if snapshot_meta is None or snapshot_meta['step']!=step:
                    raise RuntimeError('Training returned without a full-state checkpoint at its last update')
                status.update(status='complete' if complete else 'stopped',step=step,checkpoint=str(snapshot))
                write_json(folder/'status.json',status)
                print('[COMPLETE]' if complete else '[STOPPED]',method,'step='+str(step),flush=True)
        except BaseException as exc:
            status.update(status='interrupted_or_failed',error=type(exc).__name__+': '+str(exc))
            write_json(folder/'status.json',status)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    setup=sub.add_parser('prepare')
    setup.add_argument('--ha-reference',type=Path,required=True)
    setup.add_argument('--data-dir',type=Path,required=True)
    setup.add_argument('--output-root',type=Path,required=True)
    setup.add_argument('--deepspeed',type=Path,default=PROJECT.parent/'deployment/deepspeed_zero2_cpu_offload.json')
    setup.add_argument('--save-steps',type=int,default=25)
    setup.add_argument('--keep-resume',type=int,default=2)
    for name in ['preflight','run','stop','status']:
        command=sub.add_parser(name);command.add_argument('--config',type=Path,required=True)
        if name in ['run','stop']:
            command.add_argument('--method',choices=METHODS,required=True)
        if name=='run':
            command.add_argument('--stop-after-step',type=int,help='Cooperative pause at absolute optimizer step; total training schedule unchanged')
        if name=='preflight':
            command.add_argument('--report',type=Path)
    args=p.parse_args()
    if args.command=='prepare':
        prepare(args.ha_reference,args.data_dir,args.output_root,args.deepspeed,args.save_steps,args.keep_resume)
        return
    config=read(args.config)
    if args.command=='preflight':
        report=preflight(config)
        if args.report:write_json(args.report,report)
        print(json.dumps(report,indent=2));sys.exit(0 if report['ready'] else 1)
    elif args.command=='run':
        if args.stop_after_step is not None and args.stop_after_step<1:p.error('stop-after-step must be positive')
        run(config,args.method,args.stop_after_step)
    elif args.command=='stop':
        verify_config(config)
        folder=Path(config['output_root'])/args.method
        if not (folder/'run_manifest.json').is_file():
            raise ValueError('This method has not started')
        (folder/'STOP_REQUESTED').write_bytes(b'finish optimizer update, save full state, stop\n')
        print('[REQUESTED] Wait for [STOPPED]; do not kill the process during checkpoint writing.')
    else:
        output={}
        for method in config['methods']:
            folder=Path(config['output_root'])/method
            output[method]={'status':read(folder/'status.json') if (folder/'status.json').exists() else 'not_started',
                            'latest_checkpoint':read(folder/'latest_checkpoint.json') if (folder/'latest_checkpoint.json').exists() else None,
                            'selected_checkpoint':read(folder/'selected_checkpoint.json') if (folder/'selected_checkpoint.json').exists() else None}
        print(json.dumps(output,indent=2))


if __name__=='__main__':main()
