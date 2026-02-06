import torch
import torch.nn as nn

class SiglipConfig:
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

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config:SiglipConfig):
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
        
        
    
class SiglipVisionModel(nn.Module):
    def __init__(self,config:SiglipConfig):
        super().__init__()
        self.config=config
        self.vision_model=SiglipVisionTransformer(config)
        
    def forward(self,pixel_value) -> tuple:
        return self.vision_model(pixel_value=pixel_value)
    

        
        