# test_run.py
import scanpy as sc
import numpy as np
import pandas as pd

# 1. 【修改点】创建一个极小的模拟AnnData数据，完全跳过网络下载！
print("1. Creating tiny mock dataset...")

# 设置随机种子以保证结果可重现
np.random.seed(42)

# 定义数据规模：100个细胞，500个基因
n_obs, n_vars = 100, 500

# a) 生成基因表达矩阵（模拟稀疏的count数据）
# 使用负二项分布生成更接近真实scRNA-seq的数据
X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars)).astype(np.float32)

# b) 生成细胞和基因的名称
obs_names = [f'cell_{i}' for i in range(n_obs)]
var_names = [f'gene_{i}' for i in range(n_vars)]

# c) 生成空间坐标 - 这是空间转录组数据的关键！
spatial_coords = np.random.rand(n_obs, 2) * 100  # 生成100x2的矩阵，模拟二维坐标

# d) 创建AnnData对象
adata = sc.AnnData(
    X=X,
    obs=pd.DataFrame(index=obs_names),
    var=pd.DataFrame(index=var_names),
)
# 将空间坐标添加到 .obsm 中，键名必须是 'spatial'
adata.obsm['spatial'] = spatial_coords

print(f"   Created mock data shape: {adata.shape}")
print(f"   Spatial data keys: {list(adata.obsm.keys())}")
print(f"   A peek at spatial coords:\n{adata.obsm['spatial'][:5]}")

# 2. 将当前目录添加到Python路径，确保能导入你的模块
import sys
sys.path.append('.') # 这行是关键！

# 3. 尝试导入你的SAMIF类
print("2. Importing SAMIF class...")
from samif import SAMIF # 从samif.py文件中导入SAMIF类

# 4. 尝试初始化模型
print("3. Initializing model...")
# 使用极少的参数和epochs，目的只是看能否初始化，不关心训练结果
# 注意：参数名称请根据你的 __init__ 函数定义来修改 (n_epochs 或 epochs)
model = SAMIF(adata, epochs=3)

# 5. 尝试调用一个关键方法，比如.train()
print("4. Starting training (just a few steps)...")
result = model.train() # 或者可能是 .fit()，根据你的代码改

print("5. Smoke test passed! The basic structure is sound.")