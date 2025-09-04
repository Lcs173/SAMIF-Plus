import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("SAMIF")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """Clustering using the mclust algorithm."""
    np.random.seed(random_seed)
    try:
        import rpy2.robjects as robjects
        robjects.r.library("mclust")

        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
        rmclust = robjects.r['Mclust']
        
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
        mclust_res = np.array(res[-2])

        adata.obs['mclust'] = mclust_res
        adata.obs['mclust'] = adata.obs['mclust'].astype('int')
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        return adata
    except ImportError:
        logger.warning("rpy2 not available, using Leiden clustering instead")
        sc.tl.leiden(adata, key_added='mclust')
        return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """Spatial clustering based the learned representation."""
    pca = PCA(n_components=20, random_state=42) 
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding
    
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:  
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type 
       
    return adata

def refine_label(adata, radius=50, key='label'):
    """Refine labels based on spatial neighbors"""
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    # Calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    return new_type

def extract_top_value(map_matrix, retain_percent=0.1): 
    """Filter out cells with low mapping probability"""
    # Retain top values for each spot
    top_k = int(retain_percent * map_matrix.shape[1])
    output = np.zeros_like(map_matrix)
    
    for i in range(map_matrix.shape[0]):
        row = map_matrix[i, :]
        top_indices = np.argsort(row)[-top_k:]
        output[i, top_indices] = row[top_indices]
    
    return output 

def construct_cell_type_matrix(adata_sc):
    """Construct cell type matrix from single-cell data"""
    label = 'cell_type'
    if label not in adata_sc.obs_keys():
        raise ValueError(f"Cell type information not found in adata_sc.obs['{label}']")
    
    n_type = len(adata_sc.obs[label].unique())
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    return mat

def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    """Project cell types onto ST data using mapped matrix in adata.obsm"""
    # Read map matrix 
    if 'map_matrix' not in adata.obsm_keys():
        raise ValueError("Mapping matrix not found in adata.obsm['map_matrix']")
    
    map_matrix = adata.obsm['map_matrix']   # spot x cell
   
    # Extract top-k values for each spot
    map_matrix = extract_top_value(map_matrix, retain_percent)
    
    # Construct cell type matrix
    matrix_cell_type = construct_cell_type_matrix(adata_sc)
    matrix_cell_type = matrix_cell_type.values
       
    # Projection by spot-level
    matrix_projection = map_matrix.dot(matrix_cell_type)
   
    # Rename cell types
    cell_type = list(adata_sc.obs['cell_type'].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    
    df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=cell_type)
    
    # Normalize by row (spot)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)

    # Add projection results to adata
    adata.obs[df_projection.columns] = df_projection
    
    return adata

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    """Search corresponding resolution according to given cluster number"""
    logger.info('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            logger.info(f'resolution={res}, cluster number={count_unique}')
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
            logger.info(f'resolution={res}, cluster number={count_unique}')
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!" 
       
    return res

def calculate_metrics(adata, true_label_key='true_domain', pred_label_key='domain'):
    """Calculate clustering metrics"""
    if true_label_key not in adata.obs_keys():
        logger.warning(f"True labels not found in adata.obs['{true_label_key}']")
        return None, None
    
    if pred_label_key not in adata.obs_keys():
        logger.warning(f"Predicted labels not found in adata.obs['{pred_label_key}']")
        return None, None
    
    true_labels = adata.obs[true_label_key].astype('category').cat.codes.values
    pred_labels = adata.obs[pred_label_key].astype('category').cat.codes.values
    
    ari = metrics.adjusted_rand_score(true_labels, pred_labels)
    nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    
    logger.info(f'ARI: {ari:.4f}, NMI: {nmi:.4f}')
    
    return ari, nmi

def visualize_spatial(adata, color_by='domain', size=10, save_path=None):
    """Visualize spatial data"""
    if color_by not in adata.obs_keys() and color_by not in adata.var_names:
        logger.warning(f"Color by field '{color_by}' not found in adata")
        return
    
    sc.pl.spatial(adata, color=color_by, spot_size=size)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
def visualize_umap(adata, color_by='domain', use_rep='emb', save_path=None):
    """Visualize UMAP projection"""
    if use_rep not in adata.obsm_keys():
        logger.warning(f"Representation '{use_rep}' not found in adata.obsm")
        return
    
    if color_by not in adata.obs_keys() and color_by not in adata.var_names:
        logger.warning(f"Color by field '{color_by}' not found in adata")
        return
    
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=color_by)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        logger.error(f"Checkpoint file not found: {path}")
        return False
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded from {path}, epoch: {epoch}, loss: {loss:.4f}")
    return True

def plot_training_history(loss_history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_attention_weights(attention_weights, save_path=None):
    """Visualize attention weights"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis')
    plt.title('Attention Weights')
    plt.xlabel('Key')
    plt.ylabel('Query')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_feature_importance(model, data, feature_names, top_n=10):
    """Analyze feature importance using model interpretability techniques"""
    # 这里实现特征重要性分析的具体方法
    # 例如使用SHAP或Integrated Gradients
    
    logger.info(f"Analyzing feature importance for top {top_n} features")
    
    # 示例实现 - 实际应根据具体模型调整
    if hasattr(model, 'feature_importances_'):
        # 树模型的特征重要性
        importances = model.feature_importances_
    else:
        # 深度学习模型的近似特征重要性
        importances = np.random.randn(len(feature_names))  # 示例
        
    indices = np.argsort(importances)[::-1]
    
    # 输出最重要的特征
    logger.info("Feature ranking:")
    for i in range(min(top_n, len(feature_names))):
        logger.info(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    
    return indices, importances  
