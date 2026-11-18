"""Kaggle-only fresh QLoRA run; no separate evaluation campaign."""
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_dual_qlora import (require_kaggle, gpu_configuration, load_quantized_base,
    encode_completion, CompletionCollator, sha256_file, write_json, utc_now)

def main():
    require_kaggle()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(42)
    model_id='Qwen/Qwen3-4B-Instruct-2507'
    revision='cdbee75f17c01a7cc42f958dc650907174af0554'
    data=ROOT/'data/llm2_approved315_v1'
    output=ROOT/'models/llm2-approved315-v1-adapter'
    output.mkdir(parents=True,exist_ok=True)
    if any(output.glob('adapter_model.*')):
        raise RuntimeError('Refusing to overwrite an existing adapter')
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    rows=[json.loads(s) for s in (data/'sdlc/train.jsonl').read_text(encoding='utf-8').splitlines()]
    encoded=[encode_completion(row,tokenizer,max_length=4096) for row in rows]
    dataset=Dataset.from_list([{k:v for k,v in x.items() if k in ('input_ids','attention_mask','labels')} for x in encoded])
    dtype,gpu=gpu_configuration()
    manifest={'schema_version':1,'task':'sdlc','base_model':model_id,'base_revision':revision,
        'training_completed':False,'full_dataset_epochs_completed':False,'training_status':'training',
        'started_at':utc_now(),'run_tier':'full','evaluation_status':'not_requested',
        'data_sha256':{s:sha256_file(data/'sdlc'/f'{s}.jsonl') for s in ('train','test')},
        'data_kind':'User-approved synthetic teacher snapshot; not independently expert validated',
        'candidate_models':json.loads((data/'manifest.json').read_text())['candidate_models'],
        'workflow':'SRS text -> direct recommendation; no follow-up questions',
        'training_examples':len(rows),'max_sequence_tokens':max(x['length'] for x in encoded),
        'gpu':gpu,'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes')},
        'approval':json.loads((data/'manifest.json').read_text()),
        'limitations':['Synthetic teacher examples; shared blueprints; no evaluation campaign requested.']}
    write_json(output/'model_manifest.json',manifest)
    try:
        base=load_quantized_base(model_id,revision,dtype)
        base.config.use_cache=False
        base.config.pad_token_id=tokenizer.pad_token_id
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',
            task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        args=TrainingArguments(output_dir=str(output),per_device_train_batch_size=1,
            gradient_accumulation_steps=8,num_train_epochs=1,learning_rate=8e-5,
            warmup_ratio=.03,lr_scheduler_type='cosine',weight_decay=.01,max_grad_norm=.3,
            logging_steps=5,save_strategy='steps',save_steps=25,save_total_limit=2,
            eval_strategy='no',gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,tf32=False,
            optim='paged_adamw_8bit',report_to='none',remove_unused_columns=False,
            dataloader_num_workers=0,label_names=['labels'],seed=42)
        trainer=Trainer(model=model,args=args,train_dataset=dataset,
            data_collator=CompletionCollator(tokenizer.pad_token_id),processing_class=tokenizer)
        result=trainer.train()
        trainer.save_model(str(output));tokenizer.save_pretrained(str(output));trainer.save_state()
        manifest.update(training_completed=True,full_dataset_epochs_completed=True,
            training_status='completed',completed_at=utc_now(),global_step=trainer.state.global_step,
            trained_epochs=float(trainer.state.epoch),train_metrics=result.metrics,
            adapter_sha256=sha256_file(output/'adapter_model.safetensors'))
        write_json(output/'model_manifest.json',manifest)
        import shutil
        shutil.copyfile(data/'manifest.json',output/'dataset_manifest.json')
        print('Saved fresh LLM2 adapter:',output,flush=True)
    except BaseException as error:
        manifest.update(training_status='failed',failure_type=type(error).__name__,failure_message=str(error))
        write_json(output/'model_manifest.json',manifest)
        raise

if __name__=='__main__':main()
