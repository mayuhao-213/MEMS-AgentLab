# %% [markdown]
# # PINN-based Parameter Inversion for Microbeam Resonators
# 
# ---
# 
# ## 1. 数据加载与预处理
# 
# 本节实现小批量数据集的读取与基础预处理。  
# 输入为频率响应曲线（`freq`）与电流响应曲线（`m_c`），每个样本是一组 180 维的时序对（共两通道）。  
# 输出为对应的结构参数 $Q$（本数据集仅反演Q，后续可扩展为多参数）。
# 
# **注意：**  
# - 这里输入的`freq`与`m_c`均为时序曲线，不对单个样本内部flatten，也不对时刻逐点归一化，而是分别对所有样本的同一物理通道做整体标准化处理，保留完整的物理/时序结构。
# 

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# ==== 1.1 数据读取 ====
with pd.HDFStore('./data/test_small_Q.h5', 'r') as store:
    df = store['data']
    constants = store.get_storer('data').attrs.constants
    phi =  constants['phi']

print(f"已载入数据集，总样本数: {len(df)}")
print("数据集常数参数:", list(constants.keys()))

# ==== 1.2 构建输入X和输出Y ====
n_samples, T = len(df), constants['number_of_sim']
X = np.zeros((n_samples, T, 2), dtype=np.float32)
for i, row in df.iterrows():
    X[i, :, 0] = np.array(row['freq'], dtype=np.float32)
    X[i, :, 1] = np.array(row['m_c'], dtype=np.float32)
Y = df[['Q']].values.astype(np.float32)   # 这里只反演Q，后续可变多参数

# ==== 1.3 剔除全为NaN的异常样本 ====
valid_mask = ~(np.isnan(X).all(axis=(1,2)) | np.isnan(Y).any(axis=1))
X = X[valid_mask]
Y = Y[valid_mask]
print(f"剔除异常后样本数: X {X.shape}, Y {Y.shape}")

# ==== 1.4 划分训练集和测试集 ====
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"训练集: {Xtr.shape}, 测试集: {Xte.shape}")

# ==== 1.5 输入/输出标准化（每个物理通道整体归一化）====
sf = StandardScaler()
sm = StandardScaler()
st = StandardScaler()

# 对180时序的每个通道flatten成一列做标准化，再reshape回去
flat_tr_0 = Xtr[:,:,0].reshape(-1,1)
flat_te_0 = Xte[:,:,0].reshape(-1,1)
flat_tr_1 = Xtr[:,:,1].reshape(-1,1)
flat_te_1 = Xte[:,:,1].reshape(-1,1)
flat_tr_0 = sf.fit_transform(flat_tr_0)
flat_te_0 = sf.transform(flat_te_0)
flat_tr_1 = sm.fit_transform(flat_tr_1)
flat_te_1 = sm.transform(flat_te_1)
Xtr[:,:,0] = flat_tr_0.reshape(Xtr.shape[0], T)
Xtr[:,:,1] = flat_tr_1.reshape(Xtr.shape[0], T)
Xte[:,:,0] = flat_te_0.reshape(Xte.shape[0], T)
Xte[:,:,1] = flat_te_1.reshape(Xte.shape[0], T)

Ytr = st.fit_transform(Ytr)
Yte = st.transform(Yte)

# ==== 1.6 构造PyTorch数据加载器 ====
tr_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
    batch_size=32, shuffle=True)
te_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Yte)),
    batch_size=32)

print("数据加载与标准化完成。训练集与测试集已准备好。")


# %% [markdown]
# ## 2. 数据集基本信息与可视化
# 
# 本节对已加载的频响数据（freq, m_c）及目标参数（Q）进行快速可视化和基本统计。  
# 这样可以直观了解数据的形状、数值范围、异常点，以及不同样本的曲线变化趋势，有助于后续网络设计和调试。
# 
# - 打印样本数和输入输出的shape
# - 展示全数据的最小/最大数值
# - 随机抽取样本，画出对应的`freq`和`m_c`时序曲线
# - 目标Q的分布直方图
# 

# %%
# 数据基本信息
print(f"Total samples: {X.shape[0]}")
print(f"Input X shape: {X.shape}, Target Y shape: {Y.shape}")
print(f"Each input sample: freq shape {X[0,:,0].shape}, m_c shape {X[0,:,1].shape}")

# freq 和 m_c 全数据全局数值范围
print(f"freq min: {np.nanmin(X[:,:,0]):.3f}, max: {np.nanmax(X[:,:,0]):.3f}")
print(f"m_c  min: {np.nanmin(X[:,:,1]):.3f}, max: {np.nanmax(X[:,:,1]):.3f}")
print(f"Q    min: {np.nanmin(Y):.3f}, max: {np.nanmax(Y):.3f}")

# 可视化：随机3个样本的 freq 和 m_c 曲线
import matplotlib.pyplot as plt

idxs = np.random.choice(X.shape[0], size=min(3, X.shape[0]), replace=False)
phi_deg = np.linspace(10, 170, X.shape[1])
plt.figure(figsize=(12,4))
for i, idx in enumerate(idxs):
    plt.subplot(2,3,i+1)
    plt.plot(phi_deg, X[idx,:,0])
    plt.title(f"Q = {Y[idx,0]:.0f} freq")
    plt.xlabel('Phase φ (deg)')
    plt.ylabel('freq (Hz)')
    plt.subplot(2,3,i+4)
    plt.plot(phi_deg, X[idx,:,1])
    plt.title(f"Q = {Y[idx,0]:.0f} m_c")
    plt.xlabel('Phase φ (deg)')
    plt.ylabel('m_c (nA)')
plt.tight_layout()
plt.show()



# %% [markdown]
# ## 3. 网络结构定义
# 
# 本节我们为微梁参数反演任务设计一个高效的神经网络模型。  
# 由于输入数据为多通道时序序列（每个样本为 180 个相位点的 [freq, m_c] 序列），适合采用一维卷积神经网络（1D-CNN）对整个时序进行特征提取。
# 
# ### 网络结构简介
# 
# - **输入层**：每个样本输入为 shape = (180, 2)，即180个时序点、每点两个通道。
# - **卷积编码器**：多层1D卷积（带ReLU激活）堆叠，逐步抽取时序特征。
# - **全局池化**：自适应平均池化，将所有时序特征压缩为定长向量。
# - **全连接头**：若干全连接（MLP）层，将抽取的特征映射为目标参数（本任务为Q）。
# 

# %%
import torch
import torch.nn as nn

class MicrobeamPINNNet(nn.Module):
    def __init__(self, input_channels=2, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # [batch, 64, 1]
            nn.Flatten()               # [batch, 64]
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [batch, T, 2]  (需要转成 [batch, 2, T])
        x = x.permute(0,2,1)
        features = self.encoder(x)
        out = self.head(features)
        return out

# 网络实例化
net = MicrobeamPINNNet(input_channels=2, output_dim=1)
print(net)


# %% [markdown]
# ## 4. PINN物理损失设计（含数据loss、初值、边界、PDE主方程残差）
# 
# ### 4.1 数据监督损失（Data Loss）
# 
# 用真实标签Q与神经网络预测输出的均方误差（MSE）作为主监督项：
# 
# $$
# \mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^N (Q_{\text{pred}}^{(i)} - Q_{\text{true}}^{(i)})^2
# $$
# 
# ### 4.2 PDE主方程残差损失（PDE Residual Loss）
# 
# 本部分利用微梁谐振系统的已知物理主方程，将神经网络反演得到的参数（如Q）与输入响应（freq, m_c）及系统常数共同带入动力学方程，对所有时序采样点计算残差并作为损失项。
# 
# #### 物理主方程
# 
# 系统的动力学行为由如下两条非线性代数方程描述：
# 
# $$
# \begin{cases}
# - M \omega^2 y + (k_t - k_e) y + (k_{t3} - k_{e3}) \dfrac{3}{4} y^3 - F_{ac} \cos(\phi) = 0 \\
# c \omega y - F_{ac} \sin(\phi) = 0
# \end{cases}
# $$
# 
# 其中：
# - $M$：等效质量
# - $k_t, k_{t3}$：等效线性/三次刚度
# - $k_e, k_{e3}$：电-机械耦合刚度
# - $F_{ac}$：激励幅值，$F_{ac} = V_{ac} \cdot \text{trans\_factor}$
# - $c$：阻尼系数，$c = \sqrt{M k_t} / Q$
# - $\omega$：角频率（由观测freq换算：$\omega = 2\pi\,\text{freq}$）
# - $y$：响应幅值（可由m_c等反推）
# - $\phi$：激励相位
# 
# #### PDE残差定义
# 
# 将网络预测参数（如Q）与输入时序样本带入上式，  
# 对每个采样点计算两个主方程的残差，分别记为$R_1$和$R_2$：
# 
# $$
# \begin{align*}
# R_1 &= - M \omega^2 y + (k_t - k_e) y + (k_{t3} - k_{e3}) \dfrac{3}{4} y^3 - F_{ac} \cos(\phi) \\
# R_2 &= c \omega y - F_{ac} \sin(\phi)
# \end{align*}
# $$
# 
# PDE残差损失取为全时序残差的均方和：
# 
# $$
# \mathcal{L}_{pde} = \frac{1}{N T} \sum_{i=1}^N \sum_{t=1}^T \left( R_1^{(i,t)}{}^2 + R_2^{(i,t)}{}^2 \right)
# $$
# 
# - $N$为样本数，$T$为时序长度
# - $R_1^{(i,t)}$、$R_2^{(i,t)}$为第$i$个样本第$t$个采样点的方程残差
# 
# 

# %%
import torch
import torch.nn as nn

# 4.1 数据监督损失（Data Loss）：MSE
def data_loss(pred, target):
    """
    计算监督损失（均方误差 MSE）
    参数:
        pred   : 网络输出 [batch, output_dim]
        target : 真实标签 [batch, output_dim]
    返回:
        标量 MSE 损失
    """
    return nn.functional.mse_loss(pred, target)

import torch

def pde_residual_loss(pred_Q, X_seq, constants, scaler, device='cpu'):
    """
    计算PINN物理主方程残差损失
    输入:
        pred_Q:      网络输出Q，shape [batch, 1]，归一化（需反归一化）
        X_seq:       输入时序 [batch, T, 2]，0: freq, 1: m_c（应是归一化前的物理值，或已反归一化）
        constants:   dict，包含所有物理常数、phi等
        scaler:      Q的scaler（用于反归一化pred_Q为物理Q）
        device:      torch设备
    返回:
        scalar loss (均方残差)
    """
    # 1. 反归一化Q
    Q = torch.from_numpy(scaler.inverse_transform(pred_Q.detach().cpu().numpy())).to(device).float()  # shape [batch, 1]
    Q = Q.squeeze(-1)  # [batch]
    batch_size, T, _ = X_seq.shape

    # 2. 提取常数参数
    Mass      = torch.tensor(constants['Mass'],      device=device).float()
    k_t       = torch.tensor(constants['k_tt'],      device=device).float()
    k_t3      = torch.tensor(constants['k_t3'],      device=device).float()
    k_e       = torch.tensor(constants['k_e'],       device=device).float()
    k_e3      = torch.tensor(constants['k_e3'],      device=device).float()
    trans_factor = torch.tensor(constants['trans_factor'], device=device).float()
    Vac_ground  = torch.tensor(constants['Vac_ground'], device=device).float()
    phi_arr   = torch.tensor(constants['phi'],       device=device).float()  # shape [T]
    # 若Vac为时变，可用X_seq[:,:,1]自身，视具体实验

    # 3. 提取输入的freq, m_c
    freq = X_seq[:,:,0]   # [batch, T]
    m_c  = X_seq[:,:,1]   # [batch, T]
    omega = 2 * torch.pi * freq    # [batch, T]

    # 4. 由m_c近似反推y，或直接用m_c定义为y（如你的物理公式中一致）
    # 这里我们用 y = m_c / (omega * trans_factor / 1e-9) 的反推关系
    y = m_c / (omega * trans_factor / 1e-9 + 1e-12)  # 防止除零

    # 5. 其他参数
    F_ac = Vac_ground * trans_factor
    # 阻尼c与Q相关：c = sqrt(Mass * k_t) / Q
    c = torch.sqrt(Mass * k_t) / Q.view(-1,1)   # [batch, 1]，自动广播

    # 6. 计算R1和R2
    # 公式同markdown解释
    R1 = (
        -Mass * omega**2 * y
        + (k_t - k_e) * y
        + (k_t3 - k_e3) * 0.75 * y**3
        - F_ac * torch.cos(phi_arr)
    )
    R2 = (
        c * omega * y
        - F_ac * torch.sin(phi_arr)
    )

    # 7. 总物理损失（均方和）
    pde_loss = (R1**2 + R2**2).mean()
    return pde_loss

def total_loss(
    pred,         # 网络输出Q [batch, 1]，归一化
    target,       # 标签Q [batch, 1]，归一化
    X_seq,        # 输入序列 [batch, T, 2]，原始物理量
    constants,    # 常数参数dict
    scaler,       # Q的StandardScaler
    device='cpu',
    lambda_data=1.0,
    lambda_pde=1.0
):
    """
    计算总loss：监督loss（MSE）+ PDE残差loss的加权和
    """
    loss_data = data_loss(pred, target)
    loss_pde  = pde_residual_loss(pred, X_seq, constants, scaler, device)
    loss = lambda_data * loss_data + lambda_pde * loss_pde
    return loss, loss_data, loss_pde


# %% [markdown]
# ### 4.x 初值/边界损失（Initial/Boundary Loss，保留待实现）
# 
# 在物理信息神经网络（PINN）中，初值损失与边界损失通常用于强化模型在时序两端的物理约束，具体形式如下：
# 
# #### 初值损失（Initial Condition Loss）
# 
# 通常写作：
# $$
# \mathcal{L}_{\text{init}} = \frac{1}{N} \sum_{i=1}^N \left( f_{\text{NN}}(\phi_0^{(i)}) - f_{\text{theory}}(\phi_0^{(i)}) \right)^2
# $$
# 其中：
# - $f_{\text{NN}}(\phi_0^{(i)})$ 为神经网络在序列起点 $\phi_0$ 的输出
# - $f_{\text{theory}}(\phi_0^{(i)})$ 为理论/观测或物理边界的已知值
# 
# #### 边界损失（Boundary Condition Loss）
# 
# 通常写作：
# $$
# \mathcal{L}_{\text{boundary}} = \frac{1}{N} \sum_{i=1}^N \left( f_{\text{NN}}(\phi_T^{(i)}) - f_{\text{theory}}(\phi_T^{(i)}) \right)^2
# $$
# 其中：
# - $\phi_T$ 为序列终点
# - 其他符号同上
# 
# #### 在本参数反演任务中的当前情况
# 
# 本任务中，神经网络的输出为全局参数 $Q$，输入为观测到的序列（freq, m_c）。  
# 由于网络直接预测的是参数 $Q$，而不是序列本身，因此只有在能由 $Q$ 及常数、输入等反推出起点/终点理论物理量的情况下，  
# 才能通过如下方式定义初/边值损失：
# 
# $$
# \mathcal{L}_{\text{init/boundary}} = \frac{1}{N} \sum_{i=1}^N \left( f_{\text{theory}}(\phi_0^{(i)}, Q^{(i)}) - f_{\text{obs}}(\phi_0^{(i)}) \right)^2 + \left( f_{\text{theory}}(\phi_T^{(i)}, Q^{(i)}) - f_{\text{obs}}(\phi_T^{(i)}) \right)^2
# $$
# 
# - $f_{\text{theory}}$ 表示由网络反演参数 $Q$、常数与输入推算得到的理论边界值
# - $f_{\text{obs}}$ 为真实观测边界数据
# 
# **若后续获得解析解或理论边界表达式，可据此添加该loss项实现。当前本loss项暂未实现，仅作为结构预留。**
# 

# %%



