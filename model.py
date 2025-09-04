import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import logging

logger = logging.getLogger("SAMIF")

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
        return F.normalize(global_emb, p=2, dim=1) 

class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)
        
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a

class Encoder_sparse(Module):
    """Sparse version of Encoder"""
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
         
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)       
       
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a     

class Encoder_sc(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout
        
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))
        
        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
      
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        
        # Encoder
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        
        # Decoder
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        
        return x

class Encoder_map(nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        return self.M

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(dim, dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and split into heads
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.out_linear(context)
        
        return output, attn_weights

class TransformerLayer(nn.Module):
    """Transformer layer with multi-head attention and feed-forward network"""
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x, attn_weights

class CrossModalEncoder(nn.Module):
    """Cross-modal encoder for aligning different modalities"""
    def __init__(self, dim, n_heads=4, dropout=0.1, n_layers=2):
        super(CrossModalEncoder, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Transformer layers for each modality
        self.gene_layers = nn.ModuleList([
            TransformerLayer(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        
        self.image_layers = nn.ModuleList([
            TransformerLayer(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # Cross-attention layers
        self.cross_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, gene_emb, img_emb):
        # Process each modality with its own transformer
        for i in range(self.n_layers):
            gene_emb, _ = self.gene_layers[i](gene_emb)
            img_emb, _ = self.image_layers[i](img_emb)
        
        # Cross-attention: gene attends to image
        gene_cross, _ = self.cross_attention(gene_emb, img_emb, img_emb)
        gene_emb = self.norm(gene_emb + gene_cross)
        
        # Cross-attention: image attends to gene
        img_cross, _ = self.cross_attention(img_emb, gene_emb, gene_emb)
        img_emb = self.norm(img_emb + img_cross)
        
        return gene_emb, img_emb

class DynamicFusionGate(nn.Module):
    """Dynamic gated fusion mechanism"""
    def __init__(self, dim, n_modals=2, dropout=0.1):
        super(DynamicFusionGate, self).__init__()
        self.dim = dim
        self.n_modals = n_modals
        
        # Gating mechanisms for each modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            ) for _ in range(n_modals)
        ])
        
        # Feature transformation for each modality
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_modals)
        ])
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * n_modals, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, *modalities):
        assert len(modalities) == self.n_modals, "Number of modalities doesn't match"
        
        # Apply transformations and gating
        transformed = []
        for i, modal in enumerate(modalities):
            # Transform features
            t_modal = self.transforms[i](modal)
            
            # Calculate gate values
            gate = self.gates[i](modal)
            
            # Apply gating
            gated_modal = t_modal * gate
            transformed.append(gated_modal)
        
        # Concatenate and fuse
        fused = torch.cat(transformed, dim=-1)
        fused = self.fusion(fused)
        
        return fused

class Adapter(nn.Module):
    """Adapter module for fine-tuning pre-trained models"""
    def __init__(self, dim, adapter_dim=64):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x)))

class PreTrainedGeneEncoder(nn.Module):
    """Wrapper for pre-trained gene encoder with adapter"""
    def __init__(self, dim_input, dim_output, adapter_dim=64):
        super(PreTrainedGeneEncoder, self).__init__()
        
        # 模拟预训练模型
        self.pretrained_model = nn.Sequential(
            nn.Linear(dim_input, 512),
            nn.ReLU(),
            nn.Linear(512, dim_output)
        )
        
        # 初始化预训练权重
        self._init_pretrained_weights()
        
        # 添加适配器
        self.adapter = Adapter(dim_output, adapter_dim)
        
    def _init_pretrained_weights(self):
        """Initialize with simulated pretrained weights"""
        for layer in self.pretrained_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, adapter=False):
        features = self.pretrained_model(x)
        if adapter:
            features = features + self.adapter(features)  # 残差连接
        return features

class PreTrainedImageEncoder(nn.Module):
    """Wrapper for pre-trained image encoder with adapter"""
    def __init__(self, dim_input, dim_output, adapter_dim=64):
        super(PreTrainedImageEncoder, self).__init__()
        
        # 模拟预训练模型
        self.pretrained_model = nn.Sequential(
            nn.Linear(dim_input, 512),
            nn.ReLU(),
            nn.Linear(512, dim_output)
        )
        
        # 初始化预训练权重
        self._init_pretrained_weights()
        
        # 添加适配器
        self.adapter = Adapter(dim_output, adapter_dim)
        
    def _init_pretrained_weights(self):
        """Initialize with simulated pretrained weights"""
        for layer in self.pretrained_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, adapter=False):
        features = self.pretrained_model(x)
        if adapter:
            features = features + self.adapter(features)  # 残差连接
        return features

class RealPreTrainedGeneEncoder(nn.Module):
    """Wrapper for real pre-trained gene encoder (e.g., scGPT) with adapter"""
    def __init__(self, dim_input, dim_output, adapter_dim=64):
        super(RealPreTrainedGeneEncoder, self).__init__()
        
        try:
            # 尝试加载真实的预训练模型
            from transformers import AutoModel, AutoTokenizer
            self.pretrained_model = AutoModel.from_pretrained("scgpt")
            self.tokenizer = AutoTokenizer.from_pretrained("scgpt")
            logger.info("Loaded real scGPT model")
        except:
            # 回退到模拟预训练模型
            self.pretrained_model = nn.Sequential(
                nn.Linear(dim_input, 512),
                nn.ReLU(),
                nn.Linear(512, dim_output)
            )
            logger.warning("Failed to load real scGPT, using simulated model")
        
        # 投影层，将预训练模型输出映射到目标维度
        self.projection = nn.Linear(768, dim_output)  # 假设scGPT输出维度为768
        
        # 添加适配器
        self.adapter = Adapter(dim_output, adapter_dim)
        
        # 冻结预训练模型参数
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
    
    def forward(self, x, adapter=False):
        # 预处理基因数据以适应预训练模型输入格式
        # 这里需要根据实际预训练模型的输入要求进行调整
        if hasattr(self, 'tokenizer'):
            # 使用真实tokenizer处理输入
            inputs = self.tokenizer(x, return_tensors="pt", padding=True)
            outputs = self.pretrained_model(**inputs)
            features = self.projection(outputs.last_hidden_state.mean(dim=1))
        else:
            # 使用模拟模型
            features = self.pretrained_model(x)
        
        if adapter:
            features = features + self.adapter(features)
        
        return features

class RealPreTrainedImageEncoder(nn.Module):
    """Wrapper for real pre-trained image encoder (e.g., ViT) with adapter"""
    def __init__(self, dim_input, dim_output, adapter_dim=64):
        super(RealPreTrainedImageEncoder, self).__init__()
        
        try:
            # 尝试加载真实的预训练模型
            import torchvision.models as models
            self.pretrained_model = models.vit_b_16(pretrained=True)
            self.pretrained_model.heads = nn.Identity()  # 移除分类头
            logger.info("Loaded real Vision Transformer model")
        except:
            # 回退到模拟预训练模型
            self.pretrained_model = nn.Sequential(
                nn.Linear(dim_input, 512),
                nn.ReLU(),
                nn.Linear(512, dim_output)
            )
            logger.warning("Failed to load real ViT, using simulated model")
        
        # 投影层，将预训练模型输出映射到目标维度
        self.projection = nn.Linear(1000, dim_output)  # 假设ViT输出维度为1000
        
        # 添加适配器
        self.adapter = Adapter(dim_output, adapter_dim)
        
        # 冻结预训练模型参数
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
    
    def forward(self, x, adapter=False):
        # 预处理图像数据以适应预训练模型输入格式
        # 这里需要根据实际预训练模型的输入要求进行调整
        if hasattr(self.pretrained_model, 'heads'):
            # 使用真实ViT处理输入
            features = self.pretrained_model(x)
            features = self.projection(features)
        else:
            # 使用模拟模型
            features = self.pretrained_model(x)
        
        if adapter:
            features = features + self.adapter(features)
        
        return features