
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