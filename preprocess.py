import os
import ot
import torch
import torch.nn as nn
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from sklearn.decomposition import PCA
import logging
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

logger = logging.getLogger("SAMIF")

def filter_with_overlap_gene(adata, adata_sc):
    """Filter genes to keep only overlapping genes between two datasets"""
    # Remove all-zero-valued genes
    if 'highly_variable' not in adata.var.keys():
        raise ValueError("'highly_variable' are not existed in adata!")
    else:    
        adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
        raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
        adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   

    # Refine genes so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    logger.info(f'Number of overlap genes: {len(genes)}')

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    
    return adata, adata_sc

def permutation(feature):
    """Permute features for data augmentation"""
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated 

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # Calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # Find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    # Transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    
    logger.info('Graph constructed!')

def construct_interaction_KNN(adata, n_neighbors=3):
    """Construct KNN-based interaction graph"""
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    
    # Transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    logger.info('Graph constructed!')   

def preprocess(adata, n_top_genes=3000):
    """Preprocess spatial transcriptomics data"""
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata, deconvolution=False):
    """Extract features from data"""
    if deconvolution:
        adata_Vars = adata
    else:   
        adata_Vars = adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, (csc_matrix, csr_matrix)):
        feat = adata_Vars.X.toarray()
    else:
        feat = adata_Vars.X
    
    # Data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def add_contrastive_label(adata):
    """Add contrastive learning labels"""
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    """Preprocess sparse adjacency matrix"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    

def extract_image_features(adata, patch_size=224, model_name='resnet50'):
    """
    Extract image features from H&E images using pre-trained CNN
    """
    if 'image_paths' not in adata.obs_keys() and 'image' not in adata.uns_keys():
        logger.warning("No image data found in adata.obs['image_paths'] or adata.uns['image']")
        return adata
    
    logger.info(f"Extracting image features using {model_name}...")
    
    # 加载预训练模型
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # 移除最后一层
        feature_dim = 2048
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)
        model.heads = nn.Identity()  # 移除分类头
        feature_dim = 1000
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_features = []
    
    # 处理图像数据
    if 'image_paths' in adata.obs_keys():
        # 从文件路径加载图像
        for path in adata.obs['image_paths']:
            try:
                image = Image.open(path).convert('RGB')
                image = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    features = model(image)
                    features = features.flatten().numpy()
                
                image_features.append(features)
            except Exception as e:
                logger.error(f"Error processing image {path}: {e}")
                # 添加零向量作为占位符
                image_features.append(np.zeros(feature_dim))
    elif 'image' in adata.uns_keys():
        # 从内存中的图像数据提取特征
        for i in range(adata.n_obs):
            try:
                image_data = adata.uns['image'][i]
                if isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data).convert('RGB')
                else:
                    image = Image.fromarray(image_data.toarray()).convert('RGB')
                
                image = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    features = model(image)
                    features = features.flatten().numpy()
                
                image_features.append(features)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # 添加零向量作为占位符
                image_features.append(np.zeros(feature_dim))
    
    # 转换为numpy数组
    image_features = np.array(image_features)
    
    # 归一化特征
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    
    adata.obsm['image_features'] = image_features
    logger.info(f"Image features extracted with shape: {image_features.shape}")
    
    return adata

def prepare_data(adata, datatype):
    """Prepare data for SAMIF+ model"""
    logger.info(f"Preparing {datatype} data...")
    
    # 基本预处理
    if 'highly_variable' not in adata.var.keys():
        preprocess(adata)
    
    if 'adj' not in adata.obsm.keys():
        if datatype in ['Stereo', 'Slide']:
            construct_interaction_KNN(adata)
        else:    
            construct_interaction(adata)
    
    if 'label_CSL' not in adata.obsm.keys():    
        add_contrastive_label(adata)
    
    if 'feat' not in adata.obsm.keys():
        get_feature(adata)
    
    logger.info(f"{datatype} data preparation completed")
    return adata

def validate_data(adata, deconvolution=False, datatype='10X'):
    """Validate data integrity and compatibility"""
    logger.info(f"Validating {datatype} data...")
    
    # 检查必要的数据字段
    required_obsm = ['feat', 'feat_a', 'label_CSL', 'adj', 'graph_neigh']
    for field in required_obsm:
        if field not in adata.obsm_keys():
            raise ValueError(f"Required field '{field}' not found in adata.obsm")
    
    # 检查空间坐标
    if 'spatial' not in adata.obsm_keys():
        raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")
    
    # 对于解卷积任务，检查额外的字段
    if deconvolution:
        if 'highly_variable' not in adata.var.keys():
            raise ValueError("Highly variable genes not found in adata.var")
    
    logger.info(f"{datatype} data validation passed")

def fix_seed(seed):
    """Fix random seed for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'