# JointConn-v2 方法简介

> 本文档用于概括论文 **JointConn-v2: Learning a Joint Vector Field with Diffusion Transformers for Cross-Scale Connectivity and Dual-Timestep Modeling** 的方法部分。内容以论文原方法为基础，并吸收前面对 Method 章节的逻辑修正建议，适合作为论文讲解、项目 README、代码实现说明或答辩材料的基础文本。

---

## 1. 方法要解决的问题

本文关注文本到图像生成中的 **RGB–Depth 联合建模** 与 **深度条件图像生成**。核心问题有两个：

1. **边缘和结构区域的跨模态注意力容易退化**  
   RGB 与 depth 在物体边界、遮挡关系、薄结构等位置必须高度一致，但普通跨模态注意力往往无法稳定捕捉这些结构区域，导致边界模糊、深度错位或纹理与几何不一致。

2. **RGB 分支和 Depth 分支容易过度耦合**  
   如果强制两个模态在所有 token、所有层、所有时间步都进行强交互，语义分支可能会拖拽几何分支，导致 depth 结构被纹理或语义噪声污染，生成结果不稳定。

因此，本文的核心思想不是简单增加一个 depth adapter 或把 RGB/depth 全局融合，而是把跨模态交互建模为一种 **选择性的、几何感知的通信过程**：可靠结构区域强交互，平坦或不可靠区域弱交互。

---

## 2. 总体框架

方法采用一个 **双分支 Flux-style Diffusion Transformer** 框架，包含：

- RGB branch：负责图像 latent 的生成或去噪；
- Depth branch：负责深度 latent 的生成，或在 depth-conditioned 任务中作为条件输入；
- Frozen backbone：采用 Flux 风格的 DiT 主干，包含 MM-DiT 与 P-DiT 两阶段结构；
- LoRA adapters：用于分支内部的轻量适配；
- JointConn-v2：插入 Transformer block 中，用于 RGB 与 Depth 分支之间的跨模态连接。

训练时冻结以下模块：

- VAE encoder / decoder；
- CLIP-L 与 T5 文本编码器；
- Flux-style DiT backbone 主体参数。

仅训练：

- LoRA 参数；
- JointConn-v2 参数。

这样做的目的有两个：一是保留预训练 T2I 模型的图像生成能力，二是用较小代价学习 RGB–Depth 的跨模态几何对齐。

---

## 3. 支持的任务形式

本文统一支持两类任务。

### 3.1 Joint RGB-Depth Generation

任务形式：

```text
T -> (I, D)
```

输入是文本 prompt，模型同时生成 RGB 图像与深度图。

此时：

- RGB branch 从噪声开始生成 RGB latent；
- Depth branch 从噪声开始生成 depth latent；
- 两个分支都参与训练损失；
- 两个分支之间允许双向 cross-modal communication。

### 3.2 Depth-Conditioned Image Generation

任务形式：

```text
(T, D) -> I
```

输入是文本 prompt 与给定 depth map，模型生成符合文本语义且遵循 depth 几何结构的 RGB 图像。

此时：

- RGB branch 从噪声开始生成 RGB latent；
- Depth branch 保持为 clean depth condition；
- 训练损失只监督 RGB branch；
- Depth branch 主要向 RGB branch 提供几何条件，不应被 RGB 反向污染。

在修正后的 Method 写法中，建议显式引入 branch mask：

```text
joint generation:          m_x = 1, m_y = 1
深度条件图像生成:          m_x = 1, m_y = 0
```

其中 `m_x` 和 `m_y` 表示对应分支是否参与 flow matching loss。

---

## 4. 输入编码与 Tokenization

### 4.1 RGB 与 Depth latent 编码

RGB 图像和 depth map 都通过共享 VAE encoder 编码到 latent space：

```text
z_x = Enc(I)
z_y = Enc(Rep3(D))
```

其中：

- `I` 是 RGB 图像；
- `D` 是单通道 depth map；
- `Rep3(D)` 表示将单通道 depth 复制为三通道，以适配共享 VAE encoder 的输入格式。

### 4.2 文本编码

文本 prompt 使用两个文本编码器：

```text
Y = CLIP-L(T)
T_token = T5(T)
```

二者作用不同：

- CLIP-L 提供全局语义和风格信息；
- T5 提供 token-level 的细粒度文本约束。

### 4.3 2×2 packing

RGB latent 和 depth latent 分别经过同一个 2×2 packing 操作，转换为序列 token：

```text
X_x = P_2x2(z_x)
X_y = P_2x2(z_y)
```

对于 512×512 图像，VAE latent 通常为 64×64×4。经过 2×2 packing 后：

```text
latent grid: 64 × 64 × 4
token grid:  32 × 32
token num:   N = 1024
token dim:   4C = 16
```

packing 操作是 reshape/permutation，理论上是可逆且保持向量范数的。

---

## 5. Dual-Timestep Controller

Controller 负责为两个分支分配时间步、任务状态和调制信号。

### 5.1 双时间步建模

RGB 和 Depth 分支分别有自己的时间步：

```text
t_x: RGB branch timestep
t_y: Depth branch timestep
```

这使得模型能够处理以下情况：

- 两个分支同步去噪；
- 一个分支较干净，另一个分支较 noisy；
- depth-conditioned 任务中 depth branch 始终为 clean condition。

### 5.2 half-synchronous / half-independent 采样

训练时采用混合时间步策略：

```text
50%: t_x = t_y
50%: t_x 和 t_y 独立采样
```

这种设计的意义是：

- 同步时间步有助于学习 RGB-depth 的一致性；
- 独立时间步有助于学习非对称引导，例如 clean depth 引导 noisy RGB。

### 5.3 时间嵌入

Controller 将 `(t_x, t_y)` 通过位置编码和 MLP 转换为分支时间嵌入：

```text
e_x = MLP(PE(t_x), task, mask, scale)
e_y = MLP(PE(t_y), task, mask, scale)
```

这些时间嵌入输入到对应分支的 AdaLN 中，用于调制 DiT block。

---

## 6. JointConn-v2 模块

JointConn-v2 是本文最核心的跨模态连接模块。它插入到若干 Transformer block 中，在 RGB branch 与 Depth branch 之间建立可控的信息交换。

在第 `l` 个 block，JointConn-v2 接收：

```text
H_x^l: RGB hidden tokens
H_y^l: Depth hidden tokens
t_x, t_y: branch timesteps
img_ids: token grid coordinates
E: geometry edge energy map
text context: text embeddings
```

输出：

```text
R_x^l: 注入 RGB branch 的 residual
R_y^l: 注入 Depth branch 的 residual
```

JointConn-v2 由四个关键组件构成。

---

## 7. Geometry-Aware Swap-Q Cross-Attention

### 7.1 基本思想

普通 cross-attention 对所有 token pair 一视同仁。本文认为 RGB-depth 的强交互应该主要发生在结构可靠的位置，尤其是：

- 物体边界；
- 深度突变区域；
- 几何轮廓；
- 遮挡边界。

因此，JointConn-v2 在 cross-attention logits 中加入几何 bias。

### 7.2 Swap-Q 双向跨模态注意力

RGB 从 Depth 接收信息：

```text
U_x<-y = softmax(Q_x K_y^T / sqrt(d) + B) V_y
```

Depth 从 RGB 接收信息：

```text
U_y<-x = softmax(Q_y K_x^T / sqrt(d) + B) V_x
```

其中 `B` 是几何感知 attention bias。

### 7.3 Geometric Mask Bias

原论文中的 bias 包含两部分：

```text
B = relative_position_bias + edge_energy_bias
```

其中：

- `relative_position_bias` 表示 2D 相对位置偏置；
- `edge_energy_bias` 由 depth 边缘强度构成，用于提升结构 token 的注意力权重。

在修正版 Method 中，建议采用真正 pairwise 的边缘兼容性：

```text
B_ij = Pi(p_i - p_j) + beta * E_i * E_j * K(p_i, p_j)
```

其中：

- `E_i` 和 `E_j` 是 token i、j 的边缘强度；
- `K(p_i, p_j)` 是局部空间兼容核；
- `Pi` 是 2D relative position bias。

这样可以避免原公式中 query-side row-constant bias 被 softmax 抵消的问题。

---

## 8. Content Gate

Content Gate 是 sample-level gate，用于控制当前样本整体上是否应该进行强跨模态融合。

其输入包括：

- `t_x`, `t_y`；
- 当前 block index；
- RGB hidden states 的全局池化；
- Depth hidden states 的全局池化。

形式如下：

```text
g_o = sigmoid(MLP([t_x, t_y, sg(GAP(H_x)), sg(GAP(H_y)), layer_id]))
```

其中：

- `GAP` 是 global average pooling；
- `sg` 是 stop-gradient；
- `g_o` 是 sample-level 标量或广播张量。

Content Gate 的作用是防止模型在不可靠样本或不合适时间步上过度耦合两个分支。

---

## 9. Regional Routing

Regional Routing 是 token-level 路由器，用于决定每个 spatial token 是否需要跨模态融合。

原论文使用二分类 routing：

```text
[r_x, r_y] = softmax(Conv1x1([H_x, H_y]))
```

其中：

- `r_x` 控制 depth-to-RGB 注入强度；
- `r_y` 控制 RGB-to-depth 注入强度。

但如果论文强调 “to fuse / not to fuse”，建议在实现中扩展为三类 routing：

```text
[r_x, r_y, r_0] = softmax(Router([H_x, H_y, E, t_x, t_y]))
```

其中：

- `r_x`：当前 token 允许 depth -> RGB；
- `r_y`：当前 token 允许 RGB -> depth；
- `r_0`：当前 token 不融合。

这样逻辑上更符合 “token-level spatial selection of to fuse / not to fuse”。

---

## 10. Residual Fusion

经过 Swap-Q attention 得到的跨模态信息不会直接替换原 hidden state，而是作为 residual 注入。

RGB branch residual：

```text
R_x = w_x(t_x, t_y) * g_o * lambda_x * r_x * O_x
```

Depth branch residual：

```text
R_y = w_y(t_x, t_y) * g_o * lambda_y * r_y * O_y
```

其中：

- `w_x`, `w_y`：时间调度权重；
- `g_o`：sample-level content gate；
- `lambda_x`, `lambda_y`：branch coupling scale；
- `r_x`, `r_y`：token-level routing；
- `O_x`, `O_y`：cross-attention 输出。

随后注入到对应分支：

```text
H_x^{l+1} = DiTBlock_x(H_x^l) + R_x
H_y^{l+1} = DiTBlock_y(H_y^l) + R_y
```

对于 depth-conditioned generation，建议设定：

```text
lambda_x = 1
lambda_y = 0
```

这样 depth branch 只作为条件，不被 RGB residual 更新。

---

## 11. GCM-WFM 训练目标

GCM-WFM 全称为：

```text
Gated Cross-Modal Weighted Flow Matching
```

其目标是学习 packed sequence space 中的联合 vector field。

### 11.1 Flow Matching 目标

对于每个分支 `e in {x, y}`，模型预测 vector field：

```text
v_hat_e
```

teacher velocity：

```text
tau_e = epsilon_e - z_data_e
```

损失形式：

```text
L = sum_e m_e * || sqrt(alpha_e) * (v_hat_e - tau_e) ||^2
```

其中：

- `m_e` 是 branch loss mask；
- `alpha_e` 是 token-wise diagonal loss weight。

### 11.2 alpha 权重构成

原论文中 alpha 综合以下因素：

```text
alpha_e = sg(w_e(t) * g_bar_o * r_bar_e * g_e * (1 + beta_E * E))
```

含义如下：

- `w_e(t)`：时间权重；
- `g_bar_o`：多层 Content Gate 的平均；
- `r_bar_e`：多层 Regional Routing 的平均；
- `g_e`：分支 guidance / coupling scale；
- `E`：边缘强度；
- `sg`：stop-gradient。

### 11.3 推荐的稳定实现

为了保证训练稳定和理论条件更严密，建议实现中加入 per-sample normalization 与 clipping：

```text
alpha_tilde = w_time(t) * (1 + beta * E_loss) * q_e
alpha = clip(alpha_tilde / (mean_token(alpha_tilde) + delta), alpha_min, alpha_max)
alpha = stop_gradient(alpha)
```

其中：

```text
q_e = 1
```

是理论最干净版本。

如果继续使用 gate/routing 作为 reliability signal，则建议：

```text
q_e = clip(eps_q + sg(g_bar_o * r_bar_e), q_min, q_max)
```

这时应把 gate/routing 看作 detached adaptive preconditioner，而不是严格意义上完全不改变优化目标的 unbiased estimator。

---

## 12. Edge Energy 的计算

边缘图来自 depth gradient：

```text
E = Downsample(Normalize(Clip(sqrt((Sx * D)^2 + (Sy * D)^2))))
```

其中：

- `Sx`, `Sy` 是 Sobel 或 Scharr filter；
- 输出需要与 packed token grid 严格对齐；
- E 应该 stop-gradient。

建议区分两个 edge map：

```text
E_att:  用于 attention bias
E_loss: 用于 loss reweighting
```

原因是：

- `E_loss` 可以来自训练目标 pseudo-depth，用于监督权重；
- `E_att` 如果在 joint generation 中直接使用目标 depth，会造成 target leakage；
- joint generation 推理时，当前 step 不能依赖当前 step 还未预测出的 depth edge。

因此 joint generation 推理时建议采用 causal edge update：

```text
step 1: E_att = 0
step s+1: E_att = Edge(Decode(z0_y_estimated_from_step_s))
```

---

## 13. 训练流程概括

每个 training step 可以概括为：

```text
1. 读取 RGB 图像、caption、pseudo-depth。
2. 用 VAE 编码 RGB 和 depth。
3. 随机选择任务类型：joint 或 depth-conditioned。
4. 根据任务采样 t_x, t_y，并设置 m_x, m_y, lambda_x, lambda_y。
5. 采样噪声 epsilon_x, epsilon_y。
6. 构造 noised latent z_t_x, z_t_y。
7. 计算 teacher velocity tau_x, tau_y。
8. 计算 E_att 和 E_loss。
9. 2×2 packing 成 token sequence。
10. 前向通过 frozen DiT + LoRA + JointConn-v2。
11. 收集 gate/routing，构造 alpha。
12. 计算 GCM-WFM loss。
13. 只更新 LoRA 和 JointConn-v2。
```

---

## 14. 推理流程概括

### 14.1 Joint Generation

```text
Input: text prompt T
Output: RGB image I_hat, depth D_hat

1. 编码文本。
2. 初始化 z_x 和 z_y 为 Gaussian noise。
3. 设置 E_att = 0。
4. 从 t=1 到 t=0 迭代更新两个分支。
5. 每一步预测 v_x 和 v_y。
6. 使用 Euler 或 Heun 更新 z_x, z_y。
7. 用上一轮 depth clean estimate 更新 E_att。
8. 最终 decode z_x 和 z_y。
```

### 14.2 Depth-Conditioned Image Generation

```text
Input: text prompt T, depth map D
Output: RGB image I_hat

1. 编码文本。
2. 编码 depth 为 clean depth latent z_y。
3. 固定 t_y = 0，固定 z_y。
4. 从 Gaussian noise 初始化 z_x。
5. 计算 E_att = Edge(D)，并在整个采样过程中固定。
6. 从 t=1 到 t=0 只更新 RGB branch。
7. 使用 CFG 组合 conditional / unconditional vector field。
8. 最终 decode RGB latent。
```

---

## 15. 方法贡献总结

本文方法的核心贡献可以概括为三点：

1. **JointConn-v2：几何感知的跨模态连接器**  
   通过 Swap-Q cross-attention、2D 相对位置偏置、edge energy、Content Gate 和 Regional Routing，在 RGB 与 Depth 分支之间建立选择性连接。

2. **GCM-WFM：面向结构区域的加权 Flow Matching**  
   在 packed sequence space 中学习 joint vector field，并使用时间、边缘、gate、routing 等信息构造 token-wise 权重，强化结构区域监督。

3. **参数高效的双任务统一框架**  
   冻结大部分预训练 DiT 权重，只训练 LoRA 与 JointConn-v2，同时支持 joint generation 与 depth-conditioned generation。

---

## 16. 实现时应注意的逻辑点

为避免 Method 实现与论文表述不一致，代码中建议特别注意以下点：

1. **统一 flow matching 符号**  
   建议使用 `t=1` 为噪声、`t=0` 为数据的 reverse-time convention，并确保 teacher velocity 与采样更新方向一致。

2. **depth-conditioned 时 depth branch 不应被当作生成目标**  
   设置 `m_y = 0`，并固定 `t_y = 0`。

3. **区分 E_att 与 E_loss**  
   避免在 joint generation 的 attention 中使用 clean target depth edge。

4. **alpha 需要 stop-gradient、normalize、clip**  
   否则 gate/routing 可能通过缩小权重来逃避损失。

5. **Regional Routing 如果强调 no-fusion，建议实现三分类路由**  
   二分类 softmax 只能在两个方向之间竞争，不能显式表达“不融合”。

6. **只训练 LoRA 和 JointConn-v2**  
   VAE、文本编码器、DiT backbone 主体应保持 frozen。

---

## 17. 一句话总结

JointConn-v2 将 RGB–Depth 生成中的跨模态交互从“全局强耦合”改为“几何感知的选择性通信”，并通过 GCM-WFM 在结构关键区域给予更强监督，从而在保持图像质量和文本一致性的同时提升深度边界与几何一致性。
