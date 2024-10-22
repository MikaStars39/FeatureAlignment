<div align="center">

# FeatureAlignment 

</div>

$$
\text{FeatureAlignment} = \text{Alignment} (\text{e.g. DPO}) + \text{Mechanistic Interpretability} (\text{e.g. SAE})
$$

FeatureAlignment is a tool designed to enhance the alignment of large language models (LLMs) by leveraging the power of interpretability. The core idea behind this repository is to align models through meaningful features, which are extracted from Sparse AutoEncoders, enabling a more interpretable and transparent approach to model adjustment and fine-tuning.

## 🎯 Key Highlights
- Compatible with [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens), [SAE Lens](https://github.com/jbloomAus/SAELens) and [Transformers](https://github.com/huggingface/transformers).
- Support multiple alignment methods e.g. DPO, SimPO, TDPO, ORPO.
- Pytorch Lightning + Hydra for easy training.
- Template for customizing alignment methods.



## ⚡ Quick Start

### 1. 📦 Setting Up the Environment

First things first, you'll need to set up the environment. We've got you covered!

```bash
conda env create -f environment.yml
conda activate halos
```

Problems during installation? Try this manual setup:

```bash
conda create -n fpo python=3.10.12
pip3 install numpy==1.24.3 ninja==1.11.1 packaging==23.1 
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install flash-attn==2.3.3 transformers==4.35.2 datasets hydra-core==1.3.2 wandb==0.15.3 openai==1.6.1 accelerate==0.21.0 tensor-parallel==1.2.4
```

### 2. 📊 Dataset Loading

Want to load your own dataset? Add a function to `dataloader.py` like this:

```python
def get_custom_dataset(split: str, ...):
    # Your dataset loading logic here
    return Dataset
```

---

### 3. 🧑‍💻 Creating a Custom Trainer

It's time to customize your trainer! 🎯 Here's a simple example based on KTO (Kahneman-Tversky Optimization):

```python
class CustomKTOTrainer(UnpairedPreferenceTrainer):
    def loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
        # Define the KTO loss here
        return loss
```

Want more flexibility? Extend this for your own **Human-Aware Loss Functions (HALOs)**!

---

### 4. 🚀 Training Your Model

Train your model on datasets like SHP, HH, or OpenAssistant with one simple command:

```bash
python train.py loss=kto model=llama7b datasets=[shp,hh,oasst] exp_name=my_experiment mode=train ++cache_dir=/data/models
```

Your model will be saved to `/data/models/my_experiment/LATEST/policy.pt` 🎯.

---

### 5. 🧪 Sampling and Evaluation

After training, generate some samples with your new model using:

```bash
python eval.py --config-path=config.yaml ++n_samples=512 ++model.eval_batch_size=32 ++samples_dir=samples/
```

And evaluate those samples with **GPT-4** using:

```bash
python compare.py -f samples/my_experiment.json -mc 512 -bk chosen -ck policy -r results.jsonl
```

🎉 Boom! You're now generating aligned models like a pro.

---

## 🎉 Why FPO?

💡 **Why choose FPO?**
- **Scalable** from 1B to 30B models 💪.
- Built-in support for feature-level optimization through **Sparse Autoencoders** 🤖.
- 🧠 **Human feedback** incorporated more naturally through novel loss functions.

---

## 🛠️ FAQs

1. **Multi-node training support?**
   - Not yet! Currently, we support single-node training (8 x A100 GPUs). Stay tuned! 🔜

2. **How can I save checkpoints?**
   - Add `++config.intermediate_checkpoints=true` to save intermediate checkpoints.

3. **Where can I find the models?**
   - You can find the full suite of models on Huggingface (see the table below).

---

## 📚 Citation

If you find this repo or our paper useful, please feel free to cite us:

```bibtex
TODO
```

---

Thanks for checking out our repo! 🙌 Feel free to contribute or raise any issues you encounter!
