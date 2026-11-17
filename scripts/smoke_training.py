"""CPU optimizer-step smoke test using the unmodified historical HA-ASIDE modules."""
import sys,math
from common import ROOT,write

# No experiment modules are imported before selecting the historical source tree.
historical=ROOT/'reproducibility/historical_training/ha-aside-paper'
sys.path.insert(0,str(historical))
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from ha_aside.hf_adapters import HAASIDEModelWrapper
from training.dataset import HierarchicalConversationDataset,collate_hierarchical_batch
from training.chatqa import encode_conversation

def main():
    torch.manual_seed(42);torch.set_num_threads(1)
    vocab={'<pad>':0,'<unk>':1,'<eos>':2,'user':3,'assistant':4,'system':5,
           'The':6,'river':7,'is':8,'Thames':9,'Name':10,'?':11,'.':12}
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',unk_token='<unk>',eos_token='<eos>')
    tokenizer.chat_template="{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}"
    config=Qwen2Config(vocab_size=len(vocab),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=256,
                       pad_token_id=0,eos_token_id=2,attention_dropout=0.0)
    base=Qwen2ForCausalLM(config);base.config.use_cache=False
    wrapper=HAASIDEModelWrapper(base,mode='ha_aside',enable_rgag=False)
    row={'document':'The river is Thames.','messages':[{'role':'user','content':'Name river?'}],
         'answers':['Thames.']}
    row['n_tokens']=len(encode_conversation(row,tokenizer,256)['input_ids'][0])
    ds=HierarchicalConversationDataset([row],tokenizer,max_length=256)
    batch=collate_hierarchical_batch([ds[0]],pad_token_id=0)
    before=base.get_input_embeddings().weight.detach().clone()
    optimizer=torch.optim.AdamW(wrapper.parameters(),lr=1e-3,weight_decay=0)
    loss=wrapper(**batch).loss
    if not torch.isfinite(loss):raise AssertionError('Non-finite historical training loss')
    loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in wrapper.parameters()):
        raise AssertionError('No finite nonzero training gradient')
    optimizer.step()
    if torch.equal(before,base.get_input_embeddings().weight):raise AssertionError('Optimizer did not update embeddings')
    assert wrapper.hierarchical_embedding.angles[1]==math.pi/8
    assert not wrapper.enable_rgag
    result={'status':'passed','model':'tiny random Qwen2 CPU model','optimizer_steps':1,
            'historical_module':str(Path(__import__('ha_aside.hf_adapters',fromlist=['__file__']).__file__).relative_to(ROOT)),
            'loss':float(loss.detach()),'finite_gradient':True,'weights_updated':True,
            'scientific_effectiveness_measured':False}
    write(ROOT/'outputs/smoke_training.json',result);print(result)

if __name__=='__main__':
    from pathlib import Path
    main()
