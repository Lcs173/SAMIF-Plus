import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os
import logging
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

from .preprocess import (
    preprocess_adj, preprocess_adj_sparse, preprocess, 
    construct_interaction, construct_interaction_KNN, 
    add_contrastive_label, get_feature, permutation, fix_seed,
    filter_with_overlap_gene, extract_image_features, prepare_data, validate_data
)
from .model import (
    Encoder, Encoder_sparse, Encoder_sc, Encoder_map,
    CrossModalEncoder, DynamicFusionGate,
    PreTrainedGeneEncoder, PreTrainedImageEncoder, Adapter,
    RealPreTrainedGeneEncoder, RealPreTrainedImageEncoder
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("samif_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SAMIF")

class SAMIF:
    def __init__(self, 
        adata,
        adata_sc=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rate=0.001,
        learning_rate_sc=0.01,
        learning_rate_adapter=0.0001,
        weight_decay=0.001,
        epochs=600,
        epochs_sc=300,
        epochs_adapter=100,
        dim_output=64,
        random_seed=42,
        alpha=10,
        beta=1,
        theta=0.1,
        lamda1=10,
        lamda2=1,
        deconvolution=False,
        datatype='10X',
        use_pretrained=True,
        n_heads=4,
        fusion_type='dynamic_gate',
        # 新增参数
        adapter_dim=64,
        temperature=0.1,
        n_negatives=100,
        fusion_dropout=0.1,
        attention_dropout=0.1,
        use_batch_norm=True,
        image_model_name='resnet50',
        image_patch_size=224,
        gene_encoder_layers=3,
        gene_encoder_hidden=256,
        optimizer_type='adam',
        momentum=0.9,
        use_scheduler=True,
        scheduler_type='cosine',
        warmup_epochs=10,
        grad_clip=1.0,
        checkpoint_dir='checkpoints',
        log_interval=50
    ):
        # 设置设备
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 存储所有参数
        self.params = locals()
        del self.params['self']
        del self.params['adata']
        del self.params['adata_sc']
        
        logger.info(f"Initializing SAMIF with parameters: {self.params}")
        
        # 数据准备和验证
        self.adata = prepare_data(adata.copy(), datatype)
        validate_data(self.adata, deconvolution, datatype)
        
        if deconvolution and adata_sc is not None:
            self.adata_sc = prepare_data(adata_sc.copy(), 'scRNA')
            validate_data(self.adata_sc, True, 'scRNA')
            self.adata, self.adata_sc = filter_with_overlap_gene(self.adata, self.adata_sc)
        else:
            self.adata_sc = None
        
        # 设置随机种子
        fix_seed(random_seed)
        
        # 提取图像特征（如果可用）
        if 'image_paths' in self.adata.obs_keys() or 'image' in self.adata.uns_keys():
            self.adata = extract_image_features(
                self.adata, 
                patch_size=image_patch_size, 
                model_name=image_model_name
            )
        
        # 存储参数
        for key, value in self.params.items():
            setattr(self, key, value)
        
        # 准备特征和标签
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(
            self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])
        ).to(self.device)
        
        self.dim_input = self.features.shape[1]
        
        # 准备邻接矩阵
        if self.datatype in ['Stereo', 'Slide']:
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        # 准备图像特征（如果可用）
        if 'image_features' in self.adata.obsm.keys():
            self.image_features = torch.FloatTensor(self.adata.obsm['image_features']).to(self.device)
            self.dim_image = self.image_features.shape[1]
        else:
            self.image_features = None
            self.dim_image = 0
            
        # 解卷积任务的数据准备
        if self.deconvolution:
            if isinstance(self.adata.X, (csc_matrix, csr_matrix)):
                self.feat_sp = self.adata.X.toarray()
            else:
                self.feat_sp = self.adata.X
                
            if isinstance(self.adata_sc.X, (csc_matrix, csr_matrix)):
                self.feat_sc = self.adata_sc.X.toarray()
            else:
                self.feat_sc = self.adata_sc.X
            
            # 填充NaN为0
            self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
            self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
            
            self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
            self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
            
            self.n_cell = self.adata_sc.n_obs
            self.n_spot = self.adata.n_obs
            
        # 初始化模型
        self._init_models()
        
        # 学习率调度器
        self.schedulers = []
        
    def _init_models(self):
        """Initialize all models based on configuration"""
        logger.info("Initializing models...")
        
        # 基因表达编码器
        if self.use_pretrained:
            try:
                # 尝试使用真实预训练模型
                self.gene_encoder = RealPreTrainedGeneEncoder(
                    self.dim_input, self.dim_output, adapter_dim=self.adapter_dim
                ).to(self.device)
                logger.info("Using real pre-trained gene encoder")
            except:
                # 回退到模拟预训练模型
                self.gene_encoder = PreTrainedGeneEncoder(
                    self.dim_input, self.dim_output, adapter_dim=self.adapter_dim
                ).to(self.device)
                logger.info("Using simulated pre-trained gene encoder")
            
            # 冻结预训练权重，只训练适配器
            for param in self.gene_encoder.pretrained_model.parameters():
                param.requires_grad = False
        else:
            if self.datatype in ['Stereo', 'Slide']:
                self.gene_encoder = Encoder_sparse(
                    self.dim_input, self.dim_output, self.graph_neigh
                ).to(self.device)
            else:
                self.gene_encoder = Encoder(
                    self.dim_input, self.dim_output, self.graph_neigh
                ).to(self.device)
        
        # 图像编码器（如果图像数据可用）
        if self.image_features is not None:
            if self.use_pretrained:
                try:
                    # 尝试使用真实预训练模型
                    self.image_encoder = RealPreTrainedImageEncoder(
                        self.dim_image, self.dim_output, adapter_dim=self.adapter_dim
                    ).to(self.device)
                    logger.info("Using real pre-trained image encoder")
                except:
                    # 回退到模拟预训练模型
                    self.image_encoder = PreTrainedImageEncoder(
                        self.dim_image, self.dim_output, adapter_dim=self.adapter_dim
                    ).to(self.device)
                    logger.info("Using simulated pre-trained image encoder")
                
                # 冻结预训练权重
                for param in self.image_encoder.pretrained_model.parameters():
                    param.requires_grad = False
            else:
                self.image_encoder = nn.Sequential(
                    nn.Linear(self.dim_image, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.dim_output)
                ).to(self.device)
        
        # 跨模态编码器用于对齐
        self.cross_modal_encoder = CrossModalEncoder(
            self.dim_output, 
            n_heads=self.n_heads,
            dropout=self.attention_dropout
        ).to(self.device)
        
        # 动态融合门
        n_modals = 2 if self.image_features is not None else 1
        self.fusion_gate = DynamicFusionGate(
            self.dim_output, 
            n_modals=n_modals,
            dropout=self.fusion_dropout
        ).to(self.device)
        
        # 解卷积任务
        if self.deconvolution:
            self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
            self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)
        
        # 损失函数
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.loss_feat = nn.MSELoss()
        self.loss_contrastive = nn.CrossEntropyLoss()
        
        logger.info("Models initialized successfully")
    
    def _init_optimizers(self):
        """Initialize optimizers and schedulers"""
        optimizers = []
        
        # 跨模态对齐的优化器
        if self.use_pretrained:
            optimizers.append(optim.Adam(
                self.gene_encoder.adapter.parameters(), 
                lr=self.learning_rate_adapter,
                weight_decay=self.weight_decay
            ))
            
            if self.image_features is not None:
                optimizers.append(optim.Adam(
                    self.image_encoder.adapter.parameters(),
                    lr=self.learning_rate_adapter,
                    weight_decay=self.weight_decay
                ))
        else:
            optimizers.append(optim.Adam(
                self.gene_encoder.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            ))
            
            if self.image_features is not None:
                optimizers.append(optim.Adam(
                    self.image_encoder.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                ))
        
        optimizers.append(optim.Adam(
            self.cross_modal_encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        ))
        
        # 融合优化器
        self.fusion_optimizer = optim.Adam(
            self.fusion_gate.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 解卷积优化器
        if self.deconvolution:
            self.optimizer_sc = optim.Adam(
                self.model_sc.parameters(), 
                lr=self.learning_rate_sc
            )
            
            self.optimizer_map = optim.Adam(
                self.model_map.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # 学习率调度器
        if self.use_scheduler:
            for optimizer in optimizers:
                if self.scheduler_type == 'cosine':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.epochs_adapter
                    )
                elif self.scheduler_type == 'step':
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer, step_size=100, gamma=0.5
                    )
                else:
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', patience=10, factor=0.5
                    )
                self.schedulers.append(scheduler)
        
        return optimizers
    
    def train_cross_modal_alignment(self):
        """Train cross-modal alignment using CSD mechanism"""
        logger.info("Training cross-modal alignment...")
        
        # 初始化优化器
        optimizers = self._init_optimizers()
        
        # 训练循环
        for epoch in tqdm(range(self.epochs_adapter), desc="Cross-modal alignment"):
            # 设置模型为训练模式
            if self.use_pretrained:
                self.gene_encoder.adapter.train()
                if self.image_features is not None:
                    self.image_encoder.adapter.train()
            else:
                self.gene_encoder.train()
                if self.image_features is not None:
                    self.image_encoder.train()
                    
            self.cross_modal_encoder.train()
            
            # 前向传播 - 基因模态
            if self.use_pretrained:
                gene_emb = self.gene_encoder(self.features, adapter=True)
                gene_emb_a = self.gene_encoder(self.features_a, adapter=True)
            else:
                if self.datatype in ['Stereo', 'Slide']:
                    _, gene_emb, ret, ret_a = self.gene_encoder(
                        self.features, self.features_a, self.adj
                    )
                else:
                    _, gene_emb, ret, ret_a = self.gene_encoder(
                        self.features, self.features_a, self.adj
                    )
            
            # 前向传播 - 图像模态（如果可用）
            if self.image_features is not None:
                if self.use_pretrained:
                    img_emb = self.image_encoder(self.image_features, adapter=True)
                    # 创建增强的图像特征
                    img_features_a = permutation(self.image_features.detach().cpu().numpy())
                    img_features_a = torch.FloatTensor(img_features_a).to(self.device)
                    img_emb_a = self.image_encoder(img_features_a, adapter=True)
                else:
                    img_emb = self.image_encoder(self.image_features)
                    img_features_a = permutation(self.image_features.detach().cpu().numpy())
                    img_features_a = torch.FloatTensor(img_features_a).to(self.device)
                    img_emb_a = self.image_encoder(img_features_a)
            
            # 跨模态对齐
            if self.image_features is not None:
                # 对齐基因和图像模态
                aligned_gene, aligned_img = self.cross_modal_encoder(gene_emb, img_emb)
                aligned_gene_a, aligned_img_a = self.cross_modal_encoder(gene_emb_a, img_emb_a)
                
                # CSD损失用于基因-图像对齐
                loss_gene_img = self.csdloss(aligned_gene, aligned_img)
                loss_gene_img_a = self.csdloss(aligned_gene_a, aligned_img_a)
                
                total_loss = (loss_gene_img + loss_gene_img_a) / 2
            else:
                # 只有基因模态，使用标准对比损失
                loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
                loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
                total_loss = loss_sl_1 + loss_sl_2
            
            # 反向传播
            for optimizer in optimizers:
                optimizer.zero_grad()
                
            total_loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                for optimizer in optimizers:
                    torch.nn.utils.clip_grad_norm_(
                        optimizer.param_groups[0]['params'], 
                        self.grad_clip
                    )
            
            for optimizer in optimizers:
                optimizer.step()
            
            # 更新学习率
            if self.use_scheduler:
                for scheduler in self.schedulers:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(total_loss)
                    else:
                        scheduler.step()
            
            # 记录损失
            if epoch % self.log_interval == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
                
                # 保存检查点
                if epoch % (self.log_interval * 10) == 0:
                    self.save_checkpoint(epoch, total_loss.item(), 'alignment')
        
        logger.info("Cross-modal alignment training finished!")
    
    def csdloss(self, emb1, emb2, temperature=None):
        """改进的跨模态自蒸馏损失"""
        if temperature is None:
            temperature = self.temperature
        
        # 归一化嵌入
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        batch_size = emb1.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(emb1, emb2.t()) / temperature
        
        # 创建标签
        labels = torch.arange(batch_size).to(self.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        # 对称损失
        loss += F.cross_entropy(sim_matrix.t(), labels)
        
        return loss / 2
    
    def train_fusion(self):
        """Train the dynamic fusion gate"""
        logger.info("Training dynamic fusion gate...")
        
        # 设置模型为评估模式以进行特征提取
        if self.use_pretrained:
            self.gene_encoder.adapter.eval()
            if self.image_features is not None:
                self.image_encoder.adapter.eval()
        else:
            self.gene_encoder.eval()
            if self.image_features is not None:
                self.image_encoder.eval()
                
        self.cross_modal_encoder.eval()
        
        # 提取特征
        with torch.no_grad():
            if self.use_pretrained:
                gene_emb = self.gene_encoder(self.features, adapter=True)
            else:
                if self.datatype in ['Stereo', 'Slide']:
                    gene_emb = self.gene_encoder(self.features, self.features_a, self.adj)[1]
                else:
                    gene_emb = self.gene_encoder(self.features, self.features_a, self.adj)[1]
            
            if self.image_features is not None:
                if self.use_pretrained:
                    img_emb = self.image_encoder(self.image_features, adapter=True)
                else:
                    img_emb = self.image_encoder(self.image_features)
                
                # 对齐模态
                gene_emb, img_emb = self.cross_modal_encoder(gene_emb, img_emb)
        
        # 融合训练循环
        for epoch in tqdm(range(self.epochs), desc="Fusion training"):
            self.fusion_gate.train()
            
            # 前向传播通过融合门
            if self.image_features is not None:
                fused_emb = self.fusion_gate(gene_emb, img_emb)
            else:
                fused_emb = self.fusion_gate(gene_emb)
            
            # 重建损失
            if self.datatype in ['Stereo', 'Slide']:
                loss_recon = self.loss_feat(fused_emb, torch.spmm(self.adj, gene_emb))
            else:
                loss_recon = self.loss_feat(fused_emb, torch.mm(self.adj, gene_emb))
            
            # 对比损失
            if self.image_features is not None:
                # 使用两种模态进行对比学习
                loss_contrastive = self.contrastive_loss(fused_emb, gene_emb, img_emb)
            else:
                # 只使用基因模态
                loss_contrastive = self.contrastive_loss_single(fused_emb)
            
            total_loss = self.alpha * loss_recon + self.beta * loss_contrastive
            
            # 反向传播
            self.fusion_optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.fusion_optimizer.param_groups[0]['params'], 
                    self.grad_clip
                )
                
            self.fusion_optimizer.step()
            
            # 记录损失
            if epoch % self.log_interval == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
                
                # 保存检查点
                if epoch % (self.log_interval * 10) == 0:
                    self.save_checkpoint(epoch, total_loss.item(), 'fusion')
        
        logger.info("Fusion training finished!")
        
        # 保存融合后的嵌入
        with torch.no_grad():
            self.fusion_gate.eval()
            if self.image_features is not None:
                self.fused_emb = self.fusion_gate(gene_emb, img_emb)
            else:
                self.fused_emb = self.fusion_gate(gene_emb)
                
            self.fused_emb = self.fused_emb.detach().cpu().numpy()
            self.adata.obsm['emb'] = self.fused_emb
        
        return self.adata
    
    def contrastive_loss(self, anchor, positive, negative=None):
        """改进的对比损失函数"""
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # 正样本对相似度
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # 负样本对相似度
        if negative is not None:
            negative = F.normalize(negative, p=2, dim=1)
            neg_sim = torch.matmul(anchor, negative.t()) / self.temperature
        else:
            # 使用批次内负样本
            neg_sim = torch.matmul(anchor, anchor.t()) / self.temperature
            # 屏蔽对角线（自身）
            neg_sim = neg_sim - torch.diag(torch.diag(neg_sim))
        
        # 计算logits和labels
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long).to(self.device)
        
        return F.cross_entropy(logits, labels)
    
    def contrastive_loss_single(self, fused_emb):
        """单模态数据的对比损失"""
        return self.contrastive_loss(fused_emb, fused_emb)
    
    def train_deconvolution(self):
        """Train deconvolution model"""
        if not self.deconvolution:
            raise ValueError("Deconvolution is not enabled!")
        
        logger.info("Training deconvolution model...")
        
        # 训练scRNA编码器
        self.model_sc.train()
        
        for epoch in tqdm(range(self.epochs_sc), desc="scRNA encoder training"):
            self.model_sc.train()
            
            emb_sc = self.model_sc(self.feat_sc)
            loss = self.loss_feat(emb_sc, self.feat_sc)
            
            self.optimizer_sc.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer_sc.param_groups[0]['params'], 
                    self.grad_clip
                )
                
            self.optimizer_sc.step()
            
            # 记录损失
            if epoch % self.log_interval == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info("scRNA encoder training finished!")
        
        # 获取嵌入
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
            emb_sp = torch.FloatTensor(self.fused_emb).to(self.device)  # 使用ST数据的融合嵌入
            
            # 归一化特征
            emb_sc = F.normalize(emb_sc, p=2, dim=1)
            emb_sp = F.normalize(emb_sp, p=2, dim=1)
        
        # 训练映射矩阵
        for epoch in tqdm(range(self.epochs), desc="Mapping matrix learning"):
            self.model_map.train()
            map_matrix = self.model_map()
            
            loss_recon, loss_NCE = self.mapping_loss(map_matrix, emb_sp, emb_sc)
            loss = self.lamda1 * loss_recon + self.lamda2 * loss_NCE
            
            self.optimizer_map.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer_map.param_groups[0]['params'], 
                    self.grad_clip
                )
                
            self.optimizer_map.step()
            
            # 记录损失
            if epoch % self.log_interval == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
                # 保存检查点
                if epoch % (self.log_interval * 10) == 0:
                    self.save_checkpoint(epoch, loss.item(), 'deconvolution')
        
        logger.info("Mapping matrix learning finished!")
        
        # 获取最终映射矩阵
        with torch.no_grad():
            self.model_map.eval()
            map_matrix = F.softmax(self.model_map(), dim=1).cpu().numpy()
            
            self.adata.obsm['map_matrix'] = map_matrix.T  # spot x cell
            self.adata_sc.obsm['emb_sc'] = emb_sc.cpu().numpy()
        
        return self.adata, self.adata_sc
    
    def mapping_loss(self, map_matrix, emb_sp, emb_sc):
        """Calculate mapping loss"""
        # Cell-to-spot mapping
        map_probs = F.softmax(map_matrix, dim=1)  # dim=0: normalization by cell
        pred_sp = torch.matmul(map_probs.t(), emb_sc)
        
        # Reconstruction loss
        loss_recon = self.loss_feat(pred_sp, emb_sp)
        
        # Noise contrastive estimation loss
        loss_NCE = self.noise_contrastive_estimation(pred_sp, emb_sp)
        
        return loss_recon, loss_NCE
    
    def noise_contrastive_estimation(self, pred_sp, emb_sp):
        """Noise contrastive estimation loss"""
        mat = self.cosine_similarity(pred_sp, emb_sp)
        k = torch.exp(mat).sum(dim=1) - torch.exp(torch.diag(mat))
        
        # Positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(dim=1)
        
        ave = torch.div(p, k)
        loss = -torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):
        """Calculate cosine similarity"""
        M = torch.matmul(pred_sp, emb_sp.t())
        norm_c = torch.norm(pred_sp, p=2, dim=1)
        norm_s = torch.norm(emb_sp, p=2, dim=1)
        norm = torch.matmul(
            norm_c.reshape((pred_sp.shape[0], 1)), 
            norm_s.reshape((emb_sp.shape[0], 1)).t()
        ) + 1e-12
        M = torch.div(M, norm)
        
        # Handle NaN values
        if torch.any(torch.isnan(M)):
            M = torch.where(torch.isnan(M), torch.full_like(M, 0.0), M)
        
        return M
    
    def save_checkpoint(self, epoch, loss, stage):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'stage': stage,
            'model_state': {
                'gene_encoder': self.gene_encoder.state_dict(),
                'fusion_gate': self.fusion_gate.state_dict(),
                'cross_modal_encoder': self.cross_modal_encoder.state_dict(),
            },
            'optimizer_state': {
                'fusion_optimizer': self.fusion_optimizer.state_dict(),
            },
            'params': self.params
        }
        
        # 添加图像编码器状态（如果可用）
        if self.image_features is not None:
            checkpoint['model_state']['image_encoder'] = self.image_encoder.state_dict()
        
        # 添加解卷积模型状态（如果可用）
        if self.deconvolution:
            checkpoint['model_state']['model_sc'] = self.model_sc.state_dict()
            checkpoint['model_state']['model_map'] = self.model_map.state_dict()
            checkpoint['optimizer_state']['optimizer_sc'] = self.optimizer_sc.state_dict()
            checkpoint['optimizer_state']['optimizer_map'] = self.optimizer_map.state_dict()
        
        # 保存检查点
        filename = f"{self.checkpoint_dir}/checkpoint_{stage}_epoch_{epoch}.pt"
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename):
        """Load training checkpoint"""
        if not os.path.exists(filename):
            logger.error(f"Checkpoint file not found: {filename}")
            return False
        
        checkpoint = torch.load(filename, map_location=self.device)
        
        # 加载模型状态
        self.gene_encoder.load_state_dict(checkpoint['model_state']['gene_encoder'])
        self.fusion_gate.load_state_dict(checkpoint['model_state']['fusion_gate'])
        self.cross_modal_encoder.load_state_dict(checkpoint['model_state']['cross_modal_encoder'])
        
        # 加载图像编码器状态（如果可用）
        if 'image_encoder' in checkpoint['model_state'] and self.image_features is not None:
            self.image_encoder.load_state_dict(checkpoint['model_state']['image_encoder'])
        
        # 加载解卷积模型状态（如果可用）
        if self.deconvolution and 'model_sc' in checkpoint['model_state']:
            self.model_sc.load_state_dict(checkpoint['model_state']['model_sc'])
            self.model_map.load_state_dict(checkpoint['model_state']['model_map'])
        
        # 加载优化器状态
        self.fusion_optimizer.load_state_dict(checkpoint['optimizer_state']['fusion_optimizer'])
        
        if self.deconvolution and 'optimizer_sc' in checkpoint['optimizer_state']:
            self.optimizer_sc.load_state_dict(checkpoint['optimizer_state']['optimizer_sc'])
            self.optimizer_map.load_state_dict(checkpoint['optimizer_state']['optimizer_map'])
        
        logger.info(f"Checkpoint loaded: {filename}, epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
        return True
    
    def train(self, resume_from=None):
        """Main training function"""
        # 恢复训练（如果指定了检查点）
        if resume_from and self.load_checkpoint(resume_from):
            logger.info(f"Resuming training from checkpoint: {resume_from}")
        else:
            logger.info("Starting training from scratch")
        
        # Step 1: 跨模态对齐（只有当多模态数据存在时）
        if self.image_features is not None:
            self.train_cross_modal_alignment()
        else:
            logger.info("Single modality data, skipping cross-modal alignment")
        
        # Step 2: 动态融合
        result = self.train_fusion()
        
        # Step 3: 解卷积（如果启用）
        if self.deconvolution:
            if not hasattr(self, 'adata_sc'):
                raise ValueError("Deconvolution requires scRNA-seq data but adata_sc was not provided")
            result = self.train_deconvolution()
        
        logger.info("Training completed successfully!")
        return result