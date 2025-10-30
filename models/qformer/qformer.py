from torch import Tensor, nn
from transformers import BertConfig, BertModel
import torch

class HierarchicalXRayQformer(nn.Module):   
    def __init__(
            self,
            num_queries=32,
            hidden_dim=768,
            num_layers=6,
            num_heads=12,
            intermediate_size=3072,
            dropout=0.1 
    ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.dropout=dropout
        self.norm_layer=nn.LayerNorm(hidden_dim)

        # projection layers for each tensor of features of the image features :
        self.projc_layer_3_4 = nn.Linear(256, self.hidden_dim)
        self.projc_layer_4_4 = nn.Linear(512, self.hidden_dim)
        self.projc_layer_5_4 = nn.Linear(512, self.hidden_dim)

        # initialization for projection layers :
        nn.init.xavier_uniform_(self.projc_layer_3_4.weight) # from (-a, a) with a = 1 × √(6 / (256 + 768)) 
        nn.init.xavier_uniform_(self.projc_layer_4_4.weight)
        nn.init.xavier_uniform_(self.projc_layer_5_4.weight)

        self.cross_scale_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # queries of each layer , they will be kept seperate from the qformer model
        self.query_tokens_3 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.query_tokens_4 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.query_tokens_5 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        self.q_former_conv3_4 = self._create_qformer(
            hidden_dim, 
            num_layers, 
            num_heads, 
            intermediate_size,
            dropout
        )
        
        self.q_former_conv4_4 = self._create_qformer(
            hidden_dim, 
            num_layers, 
            num_heads, 
            intermediate_size,
            dropout
        )

        self.q_former_conv5_4 = self._create_qformer(
            hidden_dim, 
            num_layers, 
            num_heads, 
            intermediate_size,
            dropout
        )

    def _create_qformer(self, hidden_dim, num_layers, num_heads, intermediate_size, dropout):
        config = BertConfig(
            hidden_size=hidden_dim, 
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout, #  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob=dropout,
            add_cross_attention=True,
            is_decoder=False,
        )
        qformer = BertModel(config)
        return nn.ModuleDict({
            'model' : qformer,
            'norm' : nn.LayerNorm(hidden_dim)
        })

    def process_single_scale(self, features, qformer_module, proj_layer, which_conv:int):
        """
        args : 
            features : (B, C, H, W)
            qformer_module : the qformer for this scale
            proj_layer : a projection that will be applied on each vvg19 layer num_channels -> hidden_dim = 768
        outputs : 
            query embeddings : (B, num_queries, hidden_dim)
        """
        B, C, H, W = features.shape
        # (B, C, H, W) -> (B, H*W, C)
        features_flat = features.flatten(2).permute(0,2,1)
        # projection to hidden dim
        features_proj = proj_layer(features_flat) # -> (B, H*W, hidden_dim)
        query_tokens = getattr(self, f'query_tokens_{which_conv}').expand(B, -1, -1)
        attention_mask = torch.ones(
            (B, features_proj.size(1)), # sequence length is h*w and varies every cnn layer (if it is changed like normal cnns) 
            dtype=torch.long,
            device=features_proj.device
        )
        outputs = qformer_module['model'](
            inputs_embeds=query_tokens,
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
            vgg_features : List features of every batch , they will be 3 tensors for each image in the batch
        outputs:
            Merged queries for the LLM : (B, merged_queries, hidden_dim)
        """
        queries_conv_3 = self.process_single_scale(vgg_features['conv_3'], 
                            self.q_former_conv3_4,
                            self.projc_layer_3_3,
                            3
                            )
        queries_conv_4 = self.process_single_scale(vgg_features['conv_4'],
                            self.q_former_conv4_4,
                            self.projc_layer_4_4,
                            4
                            )
        queries_conv_5 = self.process_single_scale(vgg_features['conv_5'],
                            self.q_former_conv5_4,
                            self.projc_layer_5_4,
                            5
                            )
        
        # cross scale attention for all the queries
        all_queries = torch.cat([queries_conv_3, queries_conv_4, queries_conv_5], dim=1) # -> (B, 32+32+32, 768)
        integrated_queries, _ = self.cross_scale_attention(
            all_queries, all_queries, all_queries
        ) # -> (B, 96, 768) but they know eachother

        # Normalization 
        return self.norm_layer(integrated_queries)
    
