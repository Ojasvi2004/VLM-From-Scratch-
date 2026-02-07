import torch
import torch.nn as nn
from typing import Tuple, Optional


class SiglipVisionConfig:
    def __init__(self,
                 hidden_size=768
                 ,intermediate_size=3072
                 ,num_hidden_layers=12,
                 num_attention_head=12,
                 num_channels=3,
                 image_size=224,
                 patch_size=16,
                 layer_norm_eps=1e-6,
                 attention_dropouts=0.0,
                 num_image_tokens:int=None,
                **kwargs):
        super().__init__()
        self.hidden_size=hidden_size,
        self.intermediate_size=intermediate_size,
        self.num_hidden_layers=num_hidden_layers,
        self.num_attention_head=num_attention_head,
        self.num_channels=num_channels,
        self.image_size=image_size,
        self.patch_size=patch_size,
        self.layer_norm_eps=layer_norm_eps,
        self.attention_dropouts=attention_dropouts,
        self.num_image_tokens=num_image_tokens

class SiglipEncoder(nn.Module):
    def __init__(self, config=SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.layers=nn.ModuleList(
            [SiglipEncoderLayer(config=self.config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(self,input_embeds:torch.Tensor)->torch.Tensor:
        hidden_states=input_embeds
        for encoder_layers in self.layers:
            hidden_states=encoder_layers(hidden_states)
        
        return hidden_states
        
        

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        embd_size=config.hidden_size,
        
        self.embeddings=SiglipVisionEmbeddings(config)
        self.encoder=SiglipEncoder(config)
        self.post_layernorms=nn.LayerNorm(embd_size,eps=config.layer_norm_eps)  
        
    def forward(self,pixel_value:torch.Tensor)->torch.Tensor:
        hidden_states=self.embeddings(pixel_value)
        last_hidden_state=self.encoder(inputs_embds=hidden_states)
        last_hidden_state=self.post_layernorms(last_hidden_state)
        return last_hidden_state

class SiglipMLP(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.fc1=nn.Linear(config.hidden_size,config.intermediate_size)
        self.fc2=nn.Linear(config.intermediate_size,config.hidden_size)

    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        hidden_states=self.fc1(hidden_states)
        hidden_states=nn.functional.gelu(hidden_states,approximate="tanh")
        hidden_states=self.fc1(hidden_states)
        return hidden_states

class SiglipAttention(nn.Module):
    
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.embed_dim=config.hidden_size
        self.num_heads=config.num_attention_head
        self.head_dim=self.embed_dim//self.num_heads
        self.scale=self.head_dim**-0.5
        self.dropout=config.attention_dropouts
        
        self.k_proj=nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj=nn.Linear(self.embed_dim,self.embed_dim)
        self.q_proj=nn.Linear(self.embed_dim,self.embed_dim)
        self.out_proj=nn.Linear(self.embed_dim,self.embed_dim)
    
    def forward(self,hidden_states:torch.Tensor)->Tuple[torch.Tensor,Optional[torch.Tensor]]:
        batch_size,seq_len,_=hidden_states.size()
        query_states=self.q_proj(hidden_states)
        key_states=self.k_proj(hidden_states)
        value_states=self.v_proj(hidden_states)
        
        query_states=query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        key_states=key_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        value_states=value_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        
        attn_weights=(torch.matmul(query_states,key_states.transpose(2,3)*self.scale))
        
        if attn_weights.size()!=(batch_size,self.num_heads,seq_len,seq_len):
            raise ValueError(
                f"Attention weigts should be of size {(batch_size,self.num_heads,seq_len,seq_len)}  but is"
                f"{attn_weights.size()}"
            )
        attn_weights=nn.functional.softmax(attn_weights,dim=-1,dtype=torch.float32).to(query_states.dtype)
        attn_weights=nn.functional.dropout(attn_weights,p=self.dropout,training=self.training)
        attn_output=torch.matmul(attn_weights,value_states)
        
        if attn_output.size()!=(batch_size,self.num_heads,seq_len,self.num_heads):
            raise ValueError(
                f"attn_outputs shuld be of size {batch_size,self.num_heads,seq_len,self.num_heads} but is"
                f"{attn_output.size()}"
            )
        attn_output=attn_output.transpose(1,2).contiguous()
        attn_output=attn_output.reshape(batch_size,seq_len,self.embed_dim)
        attn_output=self.out_proj(attn_output)
        
        return attn_output,attn_weights
    
        
        
        
        
        
    
 
class SiglipEncoderLayer(nn.Module):
     def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim=config.hidden_size
        self.self_attn=SiglipAttention(config)
        self.layer_norm1=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp=SiglipMLP(config)
        self.layer_norm2=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        
     def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        residual=hidden_states
        hidden_states=self.layer_norm1(hidden_states)
        hidden_states,_=self.self_attn(hidden_states=hidden_states)
        hidden_states=hidden_states+residual
        residual2=hidden_states
        hidden_states=self.layer_norm2(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=hidden_states+residual2
    
    
        
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()  
        self.embed_dim=config.hidden_size
        self.image_size=config.image_size
        self.patch_size=config.patch_size
        
        self.patch_embeddings=nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=None
        )
        
        self.num_patches=(self.image_size//self.patch_size)**2
        self.num_position=self.num_patches
        self.positional_embedding=nn.Embedding(self.num_position,self.embed_dim)
        
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_position).expand((-1,1)),
            persistent=False
        )
        
    def forward(self,pixel_value:torch.FloatTensor)->torch.tensor:
        _,_,height,width=pixel_value.shape  #Btach_size,channel,H,W
        patch_embd=self.patch_embeddings(pixel_value)
        embedding=patch_embd.flatten(2)
        embedding=embedding.transpose(1,2)
        embedding=embedding+self.positional_embedding(self.position_ids)
        return embedding
        
        
        
        
        
           
    
class SiglipVisionModel(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.vision_model=SiglipVisionTransformer(config)
        
    def forward(self,pixel_value) -> tuple:
        return self.vision_model(pixel_value=pixel_value)
    

        
        