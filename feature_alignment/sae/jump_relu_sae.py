import torch.nn as nn
import torch
from huggingface_hub import hf_hub_download
import numpy as np

class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon

def load_jump_relu_sae(config):
  path_to_params = hf_hub_download(
      repo_id=config.sae.sae_name_or_path,
      filename=config.sae.filename,
      force_download=False,
  )

  params = np.load(path_to_params)
  pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
  sae_model = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
  sae_model.load_state_dict(pt_params)

  if not config.sae.encoder:
    sae_model.W_enc = None
    sae_model.b_enc = None
  if not config.sae.decoder:
    sae_model.W_dec = None
    sae_model.b_dec = None
    
  return sae_model
