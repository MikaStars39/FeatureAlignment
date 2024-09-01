Some hints:
* use debug=true when debugging, avoid wandb logging
* These commands use AdamW instead of RMSprop that is originally used by Halos

SFT gemma-2-2b
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py loss=sft model=gemma-2-2b datasets='[ultrachatsft]' exp_name=sft-gemma-2-2b mode=train debug=false model.batch_size=16 model.gradient_accumulation_steps=2 model.use_flash_attention=true lr=5e-6 optimizer=AdamW
```
Note: no intermediate checkpoint


DPO gemma-2-2b
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py loss=dpo model=gemma-2-2b datasets='[ultrabin]' exp_name=dpo-gemma-2-2b mode=train debug=false model.batch_size=16 model.gradient_accumulation_steps=2 model.use_flash_attention=true lr=5e-7 optimizer=AdamW
```
Note: no intermediate checkpoint


TDPO1 gemma-2-2b
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py loss=tdpo1 model=gemma-2-2b datasets='[ultrabin]' exp_name=tdpo1-gemma-2-2b mode=train debug=false model.batch_size=4 model.eval_batch_size=8 model.gradient_accumulation_steps=8 model.use_flash_attention=true lr=5e-7 optimizer=AdamW ++model.load_from=sft-gemma-2-2b/LATEST/policy.pt
```
Note: no intermediate checkpoint
Note: Adjust eval_batch_size manually, as it is not synced with train batch size.

TDPO2 gemma-2-2b
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py loss=tdpo2 model=gemma-2-2b datasets='[ultrabin]' exp_name=tdpo1-gemma-2-2b mode=train debug=false model.batch_size=4 model.eval_batch_size=8 model.gradient_accumulation_steps=8 model.use_flash_attention=true lr=5e-7 optimizer=AdamW ++model.load_from=sft-gemma-2-2b/LATEST/policy.pt
```
Note: no intermediate checkpoint
Note: Adjust eval_batch_size manually, as it is not synced with train batch size.