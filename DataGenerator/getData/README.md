## 1. **理论推导（对照图中公式）**

原始运动方程：

$$
M\ddot{x} + c\dot{x} + kx + k_3 x^3 = F\sin(\omega t + \varphi)
$$

设解为 $x = A\sin(\omega t)$，代入得：

* $x^3 = A^3\sin^3(\omega t) = \frac{3}{4}A^3\sin(\omega t) - \frac{1}{4}A^3\sin(3\omega t)$（高次谐波可忽略主频时可省略）
* $\dot{x} = A\omega\cos(\omega t)$
* $\ddot{x} = -A\omega^2\sin(\omega t)$

**代入后，收集$\sin(\omega t)$和$\cos(\omega t)$项，得到**：

* $\sin(\omega t)$项：

  $$
  -M A \omega^2 + kA + \frac{3}{4} k_3 A^3 = F\cos\varphi
  $$
* $\cos(\omega t)$项：

  $$
  cA\omega = F\sin\varphi
  $$

---

## 2. **代码实现**

### **核心思路**

* 神经网络输出 $M, k, k\_3, c, F, \varphi, A$（其中A实际由网络推断/计算）
* 对每个样本（batch、时间点）带入上述**两个代数残差**
* loss为两项残差平方和，取均值

---

### **PyTorch实现（通用模板）**

```python
import torch
import torch.nn as nn
import math

def pde_residuals(
    M, k, k3, c, F, phi, omega, A
):
    """
    按图公式计算sin项和cos项的残差
    参数全部为 shape: (batch_size, T)
    """
    # sin项残差
    sin_res = -M * A * omega**2 + k * A + (3/4) * k3 * (A**3) - F * torch.cos(phi)
    # cos项残差
    cos_res = c * A * omega - F * torch.sin(phi)
    return sin_res, cos_res

def spring_mass_damper_pde_loss(pred, Xb, calculate_params_fn, norm_sfs, denom_clamp=1e-4, loss_mode='mse', clamp_val=1e-3):
    """
    pred: 网络输出参数 (batch, 7)（例：M, k, k3, c, F, phi, A）
    Xb: 频率信息 (batch, T, 1) 或 (batch, T)
    calculate_params_fn: 你的参数解算函数
    norm_sfs: 归一化工具
    """
    # 假设 Xb[:,:,0] 是频率点
    freq = Xb[:,:,0] if Xb.dim()==3 else Xb  # (batch, T)
    omega = freq * (2 * math.pi)

    # 网络输出还原物理参数
    # 这里示例网络直接输出所有物理量，实际可通过 calculate_params_vectorized 反归一化
    # pred: (batch, 7) → 按顺序分配参数
    M, k, k3, c, F, phi, A = [pred[:,i].unsqueeze(1) for i in range(7)] # (batch, 1) each → broadcast

    # 如果A不是网络直接输出，而是你用测量值和参数反推得到的，也可以用A = ... 计算
    # 例如: A = (mc * 1e-9) / denom
    # 具体见你的 earlier code

    # 物理残差
    sin_res, cos_res = pde_residuals(M, k, k3, c, F, phi, omega, A)
    # loss聚合
    res_all = sin_res.pow(2) + cos_res.pow(2)  # (batch, T)
    if loss_mode == 'clamp':
        loss = res_all.clamp(max=clamp_val).mean()
    else:
        loss = res_all.mean()
    return loss
```

---

## 3. **使用说明与拓展**

* 如果你的**网络只输出一部分参数，其余参数通过输入/常数/物理推导算出**，
  只需把需要优化的参数作为 `pred`，其余参数通过 `calculate_params_vectorized` 解算再传入即可。
* $A$ 可由网络直接输出，也可按你之前代码逻辑，通过输入观测 $mc$ 和参数解算
* **批量处理和自动梯度兼容**，可直接用于PyTorch训练
* **易于拓展**：可加更多高阶项或其它物理约束（比如非主谐波、其他损失等）

---

## 4. **更进一步的建议**

* **物理量归一化/反归一化**应和你的数据处理流程保持一致（即 norm\_sfs/scalers 相关部分）
* 若需引入一阶或二阶导数损失，可以补充 autograd 求导项如：

  ```python
  A_grad = torch.autograd.grad(A, omega, torch.ones_like(A), create_graph=True)[0]
  # loss += lambda_grad * (A_grad**2).mean()
  ```
* 若需要更多物理项，也可扩展 residuals 及loss聚合方式

---

## 5. **完整范例（带假定参数反归一化函数）**

```python
def calculate_params_vectorized(pred, norm_sfs):
    # 仅示例：假设 pred 归一化输出，反归一化到物理量
    # pred: (batch, 7) 对应 M, k, k3, c, F, phi, A
    # norm_sfs: scaler列表
    # 实际需根据你的数据/网络实际情况编写
    return [pred[:,i].unsqueeze(1) for i in range(7)]

def full_pde_loss(pred, Xb, norm_sfs, denom_clamp=1e-4, loss_mode='clamp', clamp_val=1e-3):
    # 反归一化参数
    M, k, k3, c, F, phi, A = calculate_params_vectorized(pred, norm_sfs)
    freq = Xb[:,:,0]
    omega = freq * (2 * math.pi)
    # 物理残差
    sin_res, cos_res = pde_residuals(M, k, k3, c, F, phi, omega, A)
    res_all = sin_res.pow(2) + cos_res.pow(2)
    if loss_mode == 'clamp':
        loss = res_all.clamp(max=clamp_val).mean()
    else:
        loss = res_all.mean()
    return loss
```

---

## 6. **小结**

* **公式严格对照手写物理推导**（sin项、cos项）
* **实现易于扩展、易于物理/参数解释**
* **可与数据loss或其它PINN约束叠加用作总loss**
