

<div align="center">
  <img src="assets/89F5EE60-13D9-416B-B395-8774B4350509.webp" alt="Llama Image" style="max-width: 200px; height: 200px; border: none;">
  <h1 style="margin: 100; padding: 30;">FeatureAlignment</h1>
</div>
<p align="center">
    <a href="https://github.com/huggingface/trl/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue"></a>
</p>
<p>
  FeatureAlignment is a tool designed to enhance the alignment of large language models (LLMs) by leveraging the power of interpretability. The core idea behind this repository is to align models through meaningful features. Traditional alignment methods in the past focused on the explicit outputs of LLMs, such as logits.
  
  In contrast, we are more interested in leveraging the inherent interpretable features of LLMs for alignment.
</p>

$$
\text{FeatureAlignment} = \text{Alignment} (\text{e.g. DPO}) + \text{Mechanistic Interpretability} (\text{e.g. SAE})
$$

## 🎯 Key Highlights
- Compatible with [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens), [SAE Lens](https://github.com/jbloomAus/SAELens) and [Transformers](https://github.com/huggingface/transformers).
- Support multiple alignment methods e.g. DPO, SimPO, TDPO, ORPO.
- Pytorch Lightning + Hydra + WandB / Neupton for easy training.
- Template for customizing alignment methods.

> [!REMINDER]
> This repository is still in a stage of rapid updates and development, and we welcome any pull requests and suggestions. If you would like to add your method to this repository, please feel free to contact us directly.

## Methods Supported

| Method | Time    | Paper                            | Official Code                                  | Support |
|--------|---------|----------------------------------|------------------------------------------------|---------|
| DPO    | 2023.05 | https://arxiv.org/abs/2305.18290 | [code](https://github.com/junkangwu/alpha-DPO) | ✅       |
| KTO    | 2024.02 | https://arxiv.org/abs/2402.01306 | [code](https://github.com/junkangwu/alpha-DPO) | TODO    |
| ORPO   | 2024.03 | https://arxiv.org/abs/2403.07691 | [code](https://github.com/junkangwu/alpha-DPO) | TODO    |
| TDPO   | 2024.04 | https://arxiv.org/abs/2404.11999 | [code](https://github.com/junkangwu/alpha-DPO) | ✅       |
| SimPO  | 2024.05 | https://arxiv.org/abs/2405.14734 | [code](https://github.com/junkangwu/alpha-DPO) | ✅       |
| α-DPO  | 2024.10 | https://arxiv.org/abs/2410.10148 | [code](https://github.com/junkangwu/alpha-DPO) | TODO    |
| FPO    | 2024.10 | -                                | [code](https://github.com/junkangwu/alpha-DPO) | ✅       |

## ⚡ Quick Start

### 1. Setting Up the Environment

First things first, you'll need to set up the environment.

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

### 2. Project Structure

Before starting training or testing, let's go over the overall structure of the project files.

```
benchmark
config
data
scripts
feature_alignment
  ├── model
  ├── sae
  ├── transformers_model
  ├── utils
train.py
test.py
```

- The `benchmark` folder stores information related to benchmarks, such as the JSON files for ArenaHard questions.
- The `config` folder contains YAML files needed to manage training parameters.
- `data` handles the processing and loading of training data.
- `feature_alignment` is the main directory containing the code for training and testing. 
  - The `sae` subdirectory includes files related to sparse autoencoder models.
  - The `model` folder contains the Lightning Module framework for training.
  - `utils` includes other general utility functions.
  - The `transformers_model` directory has Hugging Face-structured model files (e.g., `modeling_xx`) to support custom models.
- `outputs` is used to store generated outputs.
- `train.py` and `test.py` are the main entry points for training and testing.

---

### 3. Creating a Custom Dataset (if needed)

Want to load your own dataset? Add a function to dataloader.py like this:

```python
def get_custom_dataset(split: str, ...):
    # Your dataset loading logic here
    return Dataset
```

Then, add your dataset to the yaml config:

```yaml 
datasets: 
 - ultrabin
 - # [your custom dataset]
```
We support multiple datasets like SHP, HH, and Ultrachat. You can check the available datasets in the `data/dataloader.py`.

---

### 4. Creating a Custom Model (if needed)

It's time to customize your method. If you want to support a new alignment method, you can try creating your own Lightning Module for training in `feature_alignment/model/your_custom_model.py`:

```python
class CustomOModel(DPOModel):
  def a_method(self, ...):
    # Your method logic here
    return loss
  def get_batch_metrics(self, ...):
    # Your metrics logic here
    return loss, metrics
```
Please note that this is actually not "creating a model" but rather "creating a method". We recommend using the existing models as a template and replacing the method logic with your own.

---

### 5. 🚀 Training Your Model

Train your model on datasets like SHP, HH, or OpenAssistant with one simple command:

```bash
python train.py loss=sft model=llama7b
```

Override the default parameters by specifying them in the command line. 

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

---

## 📚 Citation

This project is built on top of [HALOs](https://github.com/ContextualAI/HALOs) and [Hydra-lightning](https://github.com/ashleve/lightning-hydra-template).

If you find this repo or our paper useful, please feel free to cite us:

```bibtex
TODO
```

---

