from torch import Tensor, nn
from transformers import BertConfig, BertModel
import torch

class EnhancedHierarchicalXRayQformer(nn.Module):
    def __init__(
            self,
            num_queries=32,
            hidden_dim=768,
            num_layers=6,  # Increased from 4
            num_heads=12,  # Increased from 8
            intermediate_size=3072,
            dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection layers for different feature scales
        self.proj_conv3 = nn.Linear(256, hidden_dim)
        self.proj_conv4 = nn.Linear(512, hidden_dim) 
        self.proj_conv5 = nn.Linear(512, hidden_dim)
        
        # Initialize projections properly
        nn.init.xavier_uniform_(self.proj_conv3.weight)
        nn.init.xavier_uniform_(self.proj_conv4.weight)
        nn.init.xavier_uniform_(self.proj_conv5.weight)
        
        # Enhanced Q-Former with more capacity
        self.q_former_conv3_4 = self._create_qformer(
            num_queries, hidden_dim, num_layers, num_heads, intermediate_size, dropout
        )
        
        self.q_former_conv4_4 = self._create_qformer(
            num_queries, hidden_dim, num_layers, num_heads, intermediate_size, dropout
        )

        self.q_former_conv5_4 = self._create_qformer(
            num_queries, hidden_dim, num_layers, num_heads, intermediate_size, dropout
        )
        
        # Cross-scale attention for better integration
        self.cross_scale_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _create_qformer(self, num_queries, hidden_dim, num_layers, num_heads, intermediate_size, dropout):
        query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            add_cross_attention=True,  
            is_decoder=False,
        )
        qformer = BertModel(config)
        return nn.ModuleDict({
            'query_tokens': nn.ParameterList([query_tokens]),
            'model': qformer,
            'norm': nn.LayerNorm(hidden_dim)
        })

    def process_single_scale(self, features, qformer_module, proj_layer):
        B, C, H, W = features.shape
        features_flat = features.flatten(2).permute(0, 2, 1)
        features_proj = proj_layer(features_flat)
        
        # Add activation and normalization
        features_proj = torch.relu(features_proj)
        features_proj = self.dropout(features_proj)
        
        query_tokens = qformer_module['query_tokens'][0].expand(B, -1, -1)
        attention_mask = torch.ones(
            (B, features_proj.size(1)),  
            dtype=torch.long,
            device=features_proj.device
        )
        
        outputs = qformer_module['model'](
            inputs_embeds=query_tokens,  # Fixed parameter name
            encoder_hidden_states=features_proj,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        query_output = outputs.last_hidden_state
        query_output = qformer_module['norm'](query_output)
        return query_output

    def forward(self, vgg_features): 
        # Process each scale
        queries_conv3 = self.process_single_scale(
            vgg_features['conv3_4'],
            self.q_former_conv3_4,  # Fixed attribute name
            self.proj_conv3
        )
        
        queries_conv4 = self.process_single_scale(
            vgg_features['conv4_4'],
            self.q_former_conv4_4,  # Fixed attribute name  
            self.proj_conv4
        )
        
        queries_conv5 = self.process_single_scale(
            vgg_features['conv5_4'],
            self.q_former_conv5_4,  # Fixed attribute name
            self.proj_conv5
        )
        
        # Cross-scale integration
        all_queries = torch.cat([queries_conv3, queries_conv4, queries_conv5], dim=1) # -> (B, 32+32+32, 768)
        integrated_queries, _ = self.cross_scale_attention(
            all_queries, all_queries, all_queries
        ) # -> (B, 96, 768) but they know eachother 
        
        return self.output_norm(integrated_queries)

class AdaptiveMerging(nn.Module):
    """Learnable merging instead of simple concatenation"""
    def __init__(self, hidden_dim, num_scales=3):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.weights = nn.Parameter(torch.ones(num_scales))
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, *queries):
        # Weighted combination
        weighted_queries = [w * q for w, q in zip(self.weights.softmax(dim=0), queries)]
        merged = torch.cat(weighted_queries, dim=1)
        
        # Self-attention for integration
        integrated, _ = self.attention(merged, merged, merged)
        return self.norm(integrated)