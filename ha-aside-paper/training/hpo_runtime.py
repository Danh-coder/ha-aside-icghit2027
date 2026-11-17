"""HA HPO adapters over the already-tested full-state checkpoint engine."""
from pathlib import Path

from experiments.common import write_json
from training.resumable_chatqa import ResumableTrainer,StopAtBoundary,sync_tree


class HPOStop(StopAtBoundary):
    def __init__(self,folder,root,stop_after_step=None,expected_steps=None):
        super().__init__(folder,stop_after_step)
        self.root=Path(root);self.expected_steps=expected_steps

    def wants_stop(self,state):
        return (self.root/'STOP_REQUESTED').exists() or super().wants_stop(state)

    def on_train_begin(self,args,state,control,**kwargs):
        if self.expected_steps is not None and state.max_steps!=self.expected_steps:
            raise ValueError('Trainer step count differs from the frozen HPO/warmup budget')
        return control


class HPOTrainer(ResumableTrainer):
    def __init__(self,*args,hpo_stage,**kwargs):
        self.hpo_stage=hpo_stage
        super().__init__(*args,**kwargs)

    def export_epoch(self):
        complete=self.state.global_step>=self.state.max_steps
        if self.hpo_stage!='full' and not complete:return
        super().export_epoch()
        path=Path(self.args.output_dir)/'epochs'/('checkpoint-'+str(self.state.global_step))
        if not (path/'architecture.json').exists():return
        import json
        metadata=json.loads((path/'architecture.json').read_text(encoding='utf-8'))
        provenance=metadata['provenance']
        provenance['fixed_recipe_complete']=complete and self.hpo_stage=='full'
        provenance['hpo_trial_complete']=complete
        provenance['hyperparameter_search_complete']=False
        provenance['full_grid_search_complete']=False
        write_json(path/'architecture.json',metadata)
        if complete:
            write_json(Path(self.args.output_dir)/'selected_checkpoint.json',{
                'checkpoint':str(path.resolve()),'trial':provenance['trial'],
                'selection':('fresh full-data endpoint of dev-selected multifidelity recipe' if self.hpo_stage=='full'
                             else 'subset trial endpoint; not a full-data research checkpoint'),
                'validation_loss':provenance['selected_validation_loss']})
        sync_tree(path)
