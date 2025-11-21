#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr 
from .prompt import Prompt 






 





class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = 768
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.indicator_dim = max(getattr(self.configs, 'n_indicators', 0), 0)
        self.component_dim = (2 + self.indicator_dim) if self.indicator_dim > 0 else 3
        patch_len = getattr(configs, 'patch_len', self.patch_size)
        self.patch_embedding = nn.Linear(patch_len * self.component_dim, configs.d_model)

        if configs.d_model % self.component_dim != 0:
            raise ValueError(
                f"d_model ({configs.d_model}) must be divisible by component count ({self.component_dim})."
            )
        self.per_component_dim = configs.d_model // self.component_dim
       

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        

       
        if configs.pretrained == True:
           
          
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

            
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())

        
        

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name  or 'wpe' in name:   #or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # False


       

        

        if self.task_name == 'long_term_forecast':
        
            self.out_layer = nn.Linear(
                int(self.per_component_dim * (self.patch_num + configs.prompt_length)),
                configs.pred_len
            )
            
            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight)
                    
        
            
   
            
            for layer in (self.gpt2, self.patch_embedding, self.out_layer):       
                layer.cuda()
                layer.train()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):


        if self.task_name == 'long_term_forecast':
            dec_out,res = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :],res  # [B, L, D]
        
        
        return None

   

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        

        
         
            
        indicator_dim = self.indicator_dim
        indicator_series = None
        if indicator_dim > 0:
            if x_enc.shape[-1] <= indicator_dim:
                raise ValueError(
                    "Input does not contain enough channels to extract indicator features. "
                    "Ensure indicator variables are appended to the encoder input."
                )
            indicator_series = x_enc[:, :, -indicator_dim:]
            x_enc = x_enc[:, :, :-indicator_dim]

        B, L, M = x_enc.shape
            
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
        torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        if indicator_series is not None:
            indicator_series = indicator_series - indicator_series.mean(1, keepdim=True).detach()
            indicator_series = indicator_series / (
                torch.sqrt(torch.var(indicator_series, dim=1, keepdim=True, unbiased=False) + 1e-5)
            )
 
        x = rearrange(x_enc, 'b l m -> (b m) l') 


        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
            residuals = df - trend - seasonal
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined
                
            

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c = 3)

        trend_component = x[:, 0:1, :]
        seasonal_component = x[:, 1:2, :]
        residual_component = x[:, 2:3, :]

        def patchify(component):
            comp = self.padding_patch_layer(component)
            comp = comp.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            comp = comp.permute(0, 2, 1, 3).contiguous()
            comp = comp.view(comp.shape[0], comp.shape[1], -1)
            return comp

        trend_output = patchify(trend_component)
        seasonal_output = patchify(seasonal_component)
        residual_output = patchify(residual_component)

        if indicator_dim > 0 and indicator_series is not None:
            indicator_expanded = indicator_series.unsqueeze(1).repeat(1, M, 1, 1)
            indicator_expanded = indicator_expanded.reshape(B * M, indicator_dim, L)
            indicator_output = patchify(indicator_expanded)
            final_output = torch.cat([seasonal_output, trend_output, indicator_output], dim=2)
        else:
            final_output = torch.cat([seasonal_output, trend_output, residual_output], dim=2)

        pre_prompted_embedding = self.patch_embedding(final_output.float())




            
        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

               

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B * M * self.component_dim, -1))
            
            
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=self.component_dim)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss
            

        

        
        
        outputs = outputs * stdev[:,:,:M]
        outputs = outputs + means[:,:,:M]

        return outputs,res





    










