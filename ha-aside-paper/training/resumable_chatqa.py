"""Full-state, atomically committed single-process Trainer checkpoints.

Only the new matched-baseline runner uses this module. Legacy training and RQ
generation stay unchanged. Resume snapshots contain the WHOLE wrapper, whereas
epochs/ exports retain the existing inference checkpoint format.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import uuid

import numpy as np
import torch
from transformers import Trainer, TrainerCallback

from experiments.common import digest, file_hash, write_json

COMMIT = 'resume_manifest.json'


@contextmanager
def run_lock(folder):
    """An OS lock releases even on SIGKILL; no stale PID-file lock recovery."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / '.training.lock').open('a+b')
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    except (BlockingIOError, PermissionError) as exc:
        if not locked:
            raise RuntimeError('This method already has an active training process') from exc
        raise
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def sync_tree(folder):
    for path in Path(folder).rglob('*'):
        if path.is_file():
            with path.open('r+b') as handle:
                os.fsync(handle.fileno())


def sync_directory(folder):
    if os.name != 'nt':
        descriptor = os.open(str(folder), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def inventory(folder):
    return {p.relative_to(folder).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
            for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name != COMMIT}


def validate_snapshot(path, fingerprint, *, hashes=True):
    path = Path(path)
    manifest = json.loads((path / COMMIT).read_text(encoding='utf-8'))
    if manifest['fingerprint'] != fingerprint:
        raise ValueError('Resume checkpoint belongs to another method/data/recipe/environment')
    if not (path / 'trainer_state.json').is_file() or not (path / 'rng_state.pth').is_file():
        raise ValueError('Resume requires Trainer and RNG state, not just model weights')
    state = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
    if state['global_step'] != manifest['step'] or state['max_steps'] != manifest['max_steps']:
        raise ValueError('Checkpoint step/schedule differs from its commit')
    if manifest['backend'] == 'deepspeed':
        if not list(path.glob('global_step*/*optim_states.pt')) or not list(path.glob('global_step*/*model_states.pt')):
            raise ValueError('Missing DeepSpeed optimizer/module state')
        if not (path / 'latest').is_file():
            raise ValueError('Missing DeepSpeed latest tag')
    elif not all((path / f).is_file() for f in ['model.safetensors', 'optimizer.pt', 'scheduler.pt']):
        raise ValueError('Missing model/optimizer/scheduler state')
    if hashes:
        if inventory(path) != manifest['files']:
            raise ValueError('Resume snapshot is incomplete or corrupted')
    else:
        for name, item in manifest['files'].items():
            p = (path / name).resolve()
            p.relative_to(path.resolve())
            if not p.is_file() or p.stat().st_size != item['bytes']:
                raise ValueError('Resume file size differs: ' + name)
    return manifest


def latest_snapshot(folder, fingerprint):
    """Ignore uncommitted staging; never silently restart if committed data is bad."""
    candidates = sorted((Path(folder) / 'resume').glob('checkpoint-*'),
                        key=lambda p: int(p.name.split('-')[-1]), reverse=True)
    failures = []
    for path in candidates:
        try:
            metadata = validate_snapshot(path, fingerprint)
            return path, metadata, failures
        except (OSError, ValueError, KeyError) as exc:
            failures.append({'checkpoint': str(path), 'error': str(exc)})
    if candidates:
        raise ValueError('No valid committed checkpoint remains; refusing to start again: ' + str(failures))
    return None, None, failures


class StopAtBoundary(TrainerCallback):
    """Save and stop after an optimizer update; never save half-accumulated gradients."""
    def __init__(self, folder, stop_after_step=None):
        self.folder = Path(folder)
        self.stop_after_step = stop_after_step
        self.requested_signal = None

    def wants_stop(self, state):
        return (self.requested_signal is not None or (self.folder / 'STOP_REQUESTED').exists()
                or (self.stop_after_step is not None and state.global_step >= self.stop_after_step))

    def on_step_end(self, args, state, control, **kwargs):
        if self.wants_stop(state):
            print('[STOP] Saving full state at optimizer step ' + str(state.global_step), flush=True)
            control.should_save = True
            control.should_training_stop = True
        if state.epoch is not None and abs(state.epoch-round(state.epoch)) < 1e-7:
            # Save once AFTER epoch evaluation, including its RNG and log state.
            control.should_save = False
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        complete_epoch = state.epoch is not None and abs(state.epoch - round(state.epoch)) < 1e-7
        # Trainer fires epoch_end on a partial epoch after a cooperative stop too.
        if complete_epoch:
            control.should_evaluate = True
            control.should_save = True
        else:
            control.should_evaluate = False
        return control

    @contextmanager
    def signal_handlers(self):
        old = {}
        def request(signum, frame):
            self.requested_signal = signum
            print('[STOP] Signal received; finish the optimizer update before saving.', flush=True)
        try:
            for sig in [signal.SIGINT, signal.SIGTERM]:
                old[sig] = signal.signal(sig, request)
            yield
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


class ResumableTrainer(Trainer):
    def __init__(self, *args, run_identity, provenance, export_tokenizer, keep_resume=2, **kwargs):
        self.run_identity = run_identity
        self.provenance = provenance
        self.export_tokenizer = export_tokenizer
        self.keep_resume = keep_resume
        self.staging_root = None
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1 or self.is_fsdp_enabled:
            raise ValueError('Matched recipe supports one GPU/process only')

    def _get_output_dir(self, trial):
        return str(self.staging_root) if self.staging_root else super()._get_output_dir(trial)

    def _save_rng_state(self, output_dir):
        # Same RNG content as Trainer, with NumPy encoded as builtins so that
        # weights_only=True works on the existing server's PyTorch 2.5 as well.
        numpy_state = np.random.get_state()
        payload = {'python': random.getstate(), 'cpu': torch.random.get_rng_state(),
                   'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}
        if torch.cuda.is_available():
            payload['cuda'] = torch.cuda.random.get_rng_state_all()
        torch.save(payload, Path(output_dir) / 'rng_state.pth')

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return
        # Caller validated this run's committed checksum inventory before train().
        payload = torch.load(Path(checkpoint) / 'rng_state.pth', map_location='cpu', weights_only=True)
        random.setstate(payload['python']); torch.random.set_rng_state(payload['cpu'])
        value = payload['numpy']
        np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(payload['cuda'])

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        # DS restores module + optimizer itself through Trainer's DS loader.
        # Standard state dictionaries are loaded explicitly with weights_only=True,
        # avoiding an unnecessary environment upgrade for the existing 2.5 server.
        path = Path(checkpoint)
        if not self.is_deepspeed_enabled:
            self.optimizer.load_state_dict(torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True))
        if (path / 'scheduler.pt').is_file():
            self.lr_scheduler.load_state_dict(torch.load(path / 'scheduler.pt', map_location='cpu', weights_only=True))
        expected = json.loads((path / COMMIT).read_text(encoding='utf-8'))['learning_rates']
        actual = [float(g['lr']) for g in self.optimizer.param_groups]
        if actual != expected:
            raise ValueError('Optimizer/scheduler learning rate was not restored exactly')
        print('[RESUMED] Optimizer + scheduler restored; lr=' + str(actual), flush=True)

    def _save_checkpoint(self, model, trial):
        root = Path(self.args.output_dir)
        step = self.state.global_step
        destination = root / 'resume' / ('checkpoint-' + str(step))
        if destination.exists():
            validate_snapshot(destination, self.run_identity)
            self.export_epoch()
            return
        self.staging_root = root / '.pending' / uuid.uuid4().hex
        try:
            super()._save_checkpoint(model, trial)
            staged = self.staging_root / ('checkpoint-' + str(step))
            metadata = {'format': 'matched_chatqa_resume_v1', 'fingerprint': self.run_identity,
                        'step': step, 'epoch': self.state.epoch, 'max_steps': self.state.max_steps,
                        'backend': 'deepspeed' if self.is_deepspeed_enabled else 'torch',
                        'learning_rates': [float(g['lr']) for g in self.optimizer.param_groups],
                        'files': inventory(staged)}
            write_json(staged / COMMIT, metadata)
            # Validate required DS/torch files before publishing the commit.
            validate_snapshot(staged, self.run_identity, hashes=False)
            sync_tree(staged)
            # A completed final resume commit must also have its inference export.
            self.export_epoch()
            destination.parent.mkdir(exist_ok=True)
            staged.replace(destination)
            sync_directory(destination.parent)
            write_json(root / 'latest_checkpoint.json', {'checkpoint': str(destination.resolve()), 'step': step})
            print('[COMMITTED] Full-state checkpoint: ' + str(destination), flush=True)
            # Remove only old checkpoints created in THIS run, after new commit.
            ordered = sorted(destination.parent.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
            for old in ordered[:-self.keep_resume]:
                validate_snapshot(old, self.run_identity, hashes=False)
                old.resolve().relative_to((root / 'resume').resolve())
                shutil.rmtree(old)
        finally:
            # If save fails the pending directory is deliberately retained for
            # diagnosis, and is never a resume candidate.
            self.staging_root = None

    def export_epoch(self):
        epoch = self.state.epoch
        if epoch is None or abs(epoch-round(epoch)) > 1e-7:
            return
        evaluations = [e for e in self.state.log_history if 'eval_loss' in e and e['step'] == self.state.global_step]
        if not evaluations:
            return
        root = Path(self.args.output_dir)
        destination = root / 'epochs' / ('checkpoint-' + str(self.state.global_step))
        if destination.exists():
            return
        staged = root / '.pending' / ('export-' + uuid.uuid4().hex)
        model = self.accelerator.unwrap_model(self.model)
        complete = self.state.global_step >= self.state.max_steps
        provenance = {**self.provenance, 'selected_validation_loss': evaluations[-1]['eval_loss'],
                      'fixed_recipe_complete': complete, 'hyperparameter_search_complete': False,
                      'training_step': self.state.global_step, 'training_epoch': epoch,
                      'matched_run_fingerprint': self.run_identity}
        model.save_experiment(staged, self.export_tokenizer, provenance,
                              self.accelerator.get_state_dict(self.model))
        self.state.save_to_json(str(staged / 'trainer_state.json'))
        sync_tree(staged)
        destination.parent.mkdir(exist_ok=True)
        staged.replace(destination); sync_directory(destination.parent)
        if complete:
            write_json(root / 'selected_checkpoint.json', {'checkpoint': str(destination.resolve()),
                       'selection': 'fixed three-epoch endpoint; not test-selected',
                       'validation_loss': evaluations[-1]['eval_loss'], 'trial': provenance['trial']})
        print('[EPOCH EXPORT] ' + str(destination), flush=True)
