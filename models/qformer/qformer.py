from torch import Tensor, nn
from transformers import BertConfig, BertModel
import torch

class HierarchicalXRayQformer(nn.Module):
    def __init__(
            self,
            num_queries=32,
            hidden_dim=768,
            num_layers=4,
            num_heads=8,
            intermediate_size=3072,
    ):
        super().__init__()
        self.q_former_conv3_4 = self._create_qformer(
            num_queries, 256, hidden_dim, num_layers, num_heads, intermediate_size
        )
        
        self.q_former_conv4_4 = self._create_qformer(
            num_queries, 512, hidden_dim, num_layers, num_heads, intermediate_size
        )

        self.q_former_conv5_4 = self._create_qformer(
            num_queries, 512, hidden_dim, num_layers, num_heads, intermediate_size
        )

    def _create_qformer(self, num_queries, input_dim, hidden_dim, num_layers, num_heads, intermediate_size):
        query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            add_cross_attetion=True,
            is_decoder=False,
        )
        qformer = BertModel(config)
        return nn.ModuleDict({
            'query_tokens' : nn.ParameterList([query_tokens]),
            'model' : qformer,
            'norm' : nn.LayerNorm(hidden_dim)
        })

    def process_single_scale(self, features, qformer_module, proj_layer):
        """
        args : 
            features : (B, C, H, W)
            qformer_module : the qformer for this scale
            proj_layer 
        outputs : 
            query embeddings : (B, num_queries, hidden_dim)
        """
        B, C, H, W = features.shape
        # (B, C, H, W) -> (B, H*W, C)
        features_flat = features.flatten(2).permute(0,2,1)
        # projection to hidden dim
        features_proj = proj_layer(features_flat) # -> (B, H*W, hidden_dim)
        query_tokens = qformer_module['query_tokens'][0].expand(B, -1, -1)
        attention_mask = torch.ones(
            features_proj.size[:-1],
            dtype=torch.long,
            device=features_proj.device
        )
        outputs = qformer_module['model'](
            queryembeds=query_tokens,
            encoder_hidden_states=features_proj,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        query_output = outputs.last_hidden_state
        query_output = qformer_module['norm'](query_output)

        return query_output # (B, num_queries, hidden_dim)
    
    def forward(self, vgg_features):
        """
        args: 
            vgg_features : dict with 'conv3_4', 'conv4_4', 'conv5_4'
        outputs:
            Merged queries for the LLM : (B, merged_queries, hidden_dim)
        """
        queries_conv3 = self.process_single_scale(
            vgg_features['conv3_4'],
            self.qformer_conv3,
            self.proj_conv3
        )  # (B, 32, 768)
        
        queries_conv4 = self.process_single_scale(
            vgg_features['conv4_4'],
            self.qformer_conv4,
            self.proj_conv4
        )  # (B, 32, 768)
        
        queries_conv5 = self.process_single_scale(
            vgg_features['conv5_4'],
            self.qformer_conv5,
            self.proj_conv5
        )  # (B, 32, 768)
        return queries_conv3, queries_conv4, queries_conv5

class Merging_with_concat(nn.Module):
    def forward(self, q3, q4, q5):
        merged = torch.concat([q3, q4, q5], dim=1)
        return merged
    