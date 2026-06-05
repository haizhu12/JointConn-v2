# JointConn-v2 + GCM-WFM 完整代码实现设计文档

> 面向 Codex / 自动编码代理的工程实现规范。本文档不是论文复述，而是将论文中的方法、附录实现细节，以及已修正的 Method 逻辑整理成可落地的 PyTorch 代码设计。
>
> 目标：实现一个双分支 Flux-style DiT 框架，支持：
>
> 1. 文本条件联合生成：`T -> (RGB, Depth)`；
> 2. 深度条件图像生成：`(T, Depth) -> RGB`；
> 3. 冻结 VAE、文本编码器和基础 DiT 主干，仅训练 LoRA 与 JointConn-v2；
> 4. 使用修正后的 GCM-WFM，即 branch mask + normalized/clipped/detached geometry-aware weighted flow matching。

---

## 0. 实现原则

### 0.1 必须遵守的数学约定

本实现统一采用反向时间 rectified-flow / flow-matching 约定：

```text
t = 1: pure Gaussian noise
t = 0: clean latent
```

对每个分支 `e in {x, y}`：

```text
z_t^e = (1 - t_e) * z_data^e + t_e * eps^e
teacher tau_e = eps^e - z_data^e
sampling update: z_{t-dt} = z_t - dt * v_hat
```

不要混用标准 CFM 的 `z0=noise, z1=data, tau=data-noise` 约定。

### 0.2 两个任务的 branch mask

```text
x branch = RGB branch
y branch = Depth branch
```

联合生成：

```text
m_x = 1, m_y = 1
lambda_x = 1, lambda_y = 1
t_x, t_y sampled by half-sync / half-independent policy
```

深度条件图像生成：

```text
m_x = 1, m_y = 0
lambda_x = 1, lambda_y = 0
t_x sampled
t_y = 0
z_y = clean encoded depth latent
```

含义：深度条件任务中，depth branch 是条件流，不是生成目标，不对 depth velocity 施加 loss，也不更新 depth latent。

### 0.3 几何边缘图必须拆分

使用两个边缘图：

```text
E_att  : 输入 JointConn-v2 attention bias 的边缘图
E_loss : 用于 GCM-WFM loss weight 的边缘图
```

联合生成训练时，`E_att` 不允许直接使用 clean target depth 的边缘图，以免 target leakage。默认设为 0，或使用 no-grad self-conditioning 估计。

深度条件训练/推理时，`E_att = Edge(input_depth)`，并固定不变。

`E_loss` 只用于训练 loss 权重，可以来自 pseudo-depth target，并且必须 detach。

### 0.4 GCM-WFM 中的 alpha 必须 detach、归一化、裁剪

```text
alpha = stop_gradient(clip(alpha_tilde / mean_N(alpha_tilde), alpha_min, alpha_max))
```

若使用 gate/routing 构造 `q`，只能作为 detached bounded preconditioner，不要把它实现成可通过 loss 直接反向优化的权重。

---

## 1. 推荐工程目录结构

```text
jointconn-v2/
  README.md
  pyproject.toml
  requirements.txt
  configs/
    train_jointconn_v2.yaml
    infer_joint.yaml
    infer_depth_conditioned.yaml
    eval.yaml
  scripts/
    prepare_coco_depth.py
    train.py
    infer_joint.py
    infer_depth_conditioned.py
    eval_depth_edges.py
    export_checkpoint.py
  src/
    jcv2/
      __init__.py
      config.py
      logging.py
      registry.py
      utils/
        seed.py
        dtype.py
        distributed.py
        checkpoint.py
        image_io.py
        tensor.py
      data/
        coco_dataset.py
        depth_cache.py
        transforms.py
        collate.py
      models/
        autoencoder.py
        text_encoder.py
        packing.py
        geometry.py
        controller.py
        relative_position.py
        jointconn_v2.py
        dual_flux.py
        lora_utils.py
      losses/
        gcm_wfm.py
      training/
        trainer.py
        train_state.py
        optim.py
      inference/
        schedulers.py
        cfg.py
        samplers.py
        pipelines.py
      metrics/
        edge_metrics.py
        depth_metrics.py
        fid_clip.py
      tests/
        test_packing.py
        test_geometry.py
        test_controller.py
        test_jointconn.py
        test_loss.py
        test_depth_conditioned_masks.py
```

---

## 2. 依赖设计

### 2.1 Python 版本

```text
python >= 3.10
```

### 2.2 核心依赖

建议在 `requirements.txt` 中固定主版本，避免 Diffusers 内部 Flux API 改动导致 hook 失效。

```text
torch>=2.5
torchvision>=0.20
torchaudio>=2.5
accelerate>=1.0
diffusers>=0.33
transformers>=4.45
peft>=0.13
safetensors>=0.4
huggingface_hub>=0.25
datasets>=2.20
omegaconf>=2.3
pydantic>=2.0
tqdm
pillow
opencv-python
numpy
scipy
scikit-image
torchmetrics
clean-fid
open-clip-torch
```

### 2.3 可选依赖

```text
xformers
bitsandbytes
wandb
tensorboard
```

### 2.4 外部模型接口

实现时支持两种模式：

```text
paper_mode:
  - VAE latent_channels = 4
  - 2x2 pack 后 token_dim = 16
  - 使用论文/自定义 Flux-style DiT checkpoint

hf_flux_mode:
  - 使用 Hugging Face Diffusers 的 FLUX.1 组件
  - latent_channels 由 VAE 自动读取
  - token_dim = 4 * latent_channels
  - FluxTransformer2DModel 默认输入通道通常需要与 pack 后通道对齐
```

代码必须通过配置项读取 `latent_channels` 和 `packed_channels`，不要硬编码 `C=4` 或 `C_p=16`。

---

## 3. 全局配置 Schema

在 `src/jcv2/config.py` 中定义 dataclass 或 Pydantic model。

```python
@dataclass
class DataConfig:
    dataset_name: str = "coco2017"
    image_root: str = "data/coco/train2017"
    caption_file: str = "data/coco/annotations/captions_train2017.json"
    depth_root: str = "data/coco_depth_v2/train2017"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    cache_text_embeddings: bool = False
    cache_latents: bool = False

@dataclass
class ModelConfig:
    backbone_type: str = "flux"              # flux | custom_dit
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
    mode: str = "hf_flux_mode"              # paper_mode | hf_flux_mode
    latent_channels: Optional[int] = None    # auto if None
    pack_size: int = 2
    hidden_dim: Optional[int] = None          # auto from backbone
    num_attention_heads: Optional[int] = None # auto from backbone
    connector_block_indices: list[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    connector_in_single_stream: bool = True
    freeze_vae: bool = True
    freeze_text_encoders: bool = True
    freeze_base_transformer: bool = True
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out", "ff", "proj"])

@dataclass
class JointConnConfig:
    beta_att: float = 1.0
    local_kernel_sigma: float = 3.0
    use_pairwise_edge_bias: bool = True
    routing_type: str = "three_way"          # three_way | two_sigmoid
    use_no_fusion_state: bool = True
    gate_hidden_dim: int = 512
    routing_hidden_dim: int = 256
    residual_w_max: float = 1.0
    zero_init_output_proj: bool = True
    dropout: float = 0.0

@dataclass
class LossConfig:
    beta_loss: float = 2.0
    gamma_t: float = 1.0
    alpha_min: float = 0.25
    alpha_max: float = 4.0
    alpha_eps: float = 1e-6
    use_connector_reliability_q: bool = True
    q_eps: float = 0.05
    q_min: float = 0.05
    q_max: float = 2.0
    normalize_alpha_per_sample: bool = True
    detach_alpha: bool = True

@dataclass
class TrainConfig:
    output_dir: str = "outputs/jointconn_v2"
    seed: int = 42
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"            # no | fp16 | bf16
    max_train_steps: int = 100000
    num_train_epochs: Optional[int] = None
    lr: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    p_joint_task: float = 0.5
    p_sync: float = 0.5
    time_sampling: str = "uniform"           # uniform | logit_normal | flux_shifted
    joint_train_e_att_mode: str = "zero"     # zero | self_condition_no_grad
    save_every_steps: int = 2000
    log_every_steps: int = 50
    num_workers: int = 8

@dataclass
class InferenceConfig:
    num_steps: int = 40
    solver: str = "heun"                     # euler | heun
    cfg_scale: float = 4.0
    depth_cfg_scale: float = 1.0
    width: int = 512
    height: int = 512
    joint_e_att_update: str = "causal_depth" # zero | causal_depth
    update_e_att_every: int = 1
```

---

## 4. 数据处理模块

### 4.1 数据输入

目标支持 COCO 2017：

```text
image: RGB image
caption: official English caption, train 时多 caption 随机采样，eval 时固定第一个 caption
depth: pseudo-depth generated by Depth Anything V2
```

### 4.2 深度图预生成脚本

文件：`scripts/prepare_coco_depth.py`

功能：

1. 读取 COCO image；
2. 使用 Depth Anything V2 生成相对深度；
3. 对每张图 per-image normalize 到 `[0, 1]`；
4. 保存为 `.npy` 或 16-bit PNG；
5. 保证后续训练中 image/depth 使用同一 resize/crop 参数。

接口：

```bash
python scripts/prepare_coco_depth.py \
  --image-root data/coco/train2017 \
  --out-root data/coco_depth_v2/train2017 \
  --model-name depth-anything/Depth-Anything-V2-Large-hf \
  --device cuda \
  --batch-size 4
```

推荐保存结构：

```text
data/coco_depth_v2/train2017/
  000000000009.npy
  000000000025.npy
  ...
```

### 4.3 Dataset 返回格式

文件：`src/jcv2/data/coco_dataset.py`

```python
class CocoDepthDataset(torch.utils.data.Dataset):
    def __getitem__(self, index) -> dict:
        return {
            "image": FloatTensor[3, H, W],        # RGB, range [-1, 1] or VAE preprocess range
            "depth": FloatTensor[1, H, W],        # depth, range [0, 1]
            "depth_rgb": FloatTensor[3, H, W],    # replicate depth to 3 channels, then VAE preprocess
            "caption": str,
            "image_id": str,
        }
```

注意：

- image 和 depth 必须共享同一几何变换：resize、crop、flip；
- depth 若 flip，必须同步 flip；
- depth 不应做颜色增强；
- `depth_rgb = depth.repeat(3, 1, 1)` 后再映射到 VAE 输入范围，通常为 `[-1, 1]`。

### 4.4 Transform 规则

文件：`src/jcv2/data/transforms.py`

```python
class JointImageDepthTransform:
    def __call__(self, image: PIL.Image, depth: np.ndarray) -> tuple[Tensor, Tensor]:
        # 1. resize preserving aspect ratio
        # 2. center/random crop to 512x512
        # 3. random horizontal flip with shared probability
        # 4. RGB normalize to VAE input range
        # 5. depth normalize/clamp to [0,1]
```

---

## 5. VAE 与文本编码器封装

### 5.1 VAE wrapper

文件：`src/jcv2/models/autoencoder.py`

```python
class FrozenAutoencoder(nn.Module):
    def __init__(self, vae, scaling_factor=None, shift_factor=None):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

    @torch.no_grad()
    def encode(self, image: Tensor, sample: bool = False) -> Tensor:
        """image: [B,3,H,W], output latent: [B,C,H',W']"""

    @torch.no_grad()
    def decode_rgb(self, latent: Tensor) -> Tensor:
        """latent -> RGB image tensor, range [0,1]"""

    @torch.no_grad()
    def decode_depth(self, latent: Tensor) -> Tensor:
        """latent -> single-channel normalized depth [B,1,H,W]
        实现：VAE decode 得到 3-channel image，取均值或第一通道，再 clamp/normalize 到 [0,1]。
        """
```

实现要点：

- 训练时 VAE 全程 `torch.no_grad()`；
- 如果使用 Diffusers VAE，遵守其 `scaling_factor` / `shift_factor`；
- 通过实际输出自动记录 `latent_channels`；
- 不要在 depth branch 上训练 VAE。

### 5.2 Text encoder wrapper

文件：`src/jcv2/models/text_encoder.py`

```python
class FrozenTextEncoders(nn.Module):
    def __init__(self, clip_tokenizer, clip_model, t5_tokenizer, t5_model):
        ...

    @torch.no_grad()
    def encode(self, captions: list[str]) -> dict:
        return {
            "pooled": Tensor[B, pooled_dim],       # CLIP-L global / pooled projection
            "tokens": Tensor[B, T, text_dim],      # T5 token-level states
            "text_ids": Tensor[...] | None,
            "attention_mask": Tensor[B, T] | None,
        }

    @torch.no_grad()
    def encode_uncond(self, batch_size: int) -> dict:
        return self.encode([""] * batch_size)
```

---

## 6. Packing / Unpacking

文件：`src/jcv2/models/packing.py`

### 6.1 2x2 pack

输入：`[B, C, 2h, 2w]`  
输出：`[B, h*w, 4C]`

```python
def pack_2x2(z: Tensor) -> Tensor:
    B, C, H, W = z.shape
    assert H % 2 == 0 and W % 2 == 0
    z = z.reshape(B, C, H // 2, 2, W // 2, 2)
    z = z.permute(0, 2, 4, 3, 5, 1).contiguous()
    return z.reshape(B, (H // 2) * (W // 2), 4 * C)
```

### 6.2 2x2 unpack

输入：`[B, h*w, 4C]`  
输出：`[B, C, 2h, 2w]`

```python
def unpack_2x2(tokens: Tensor, h: int, w: int, C: int) -> Tensor:
    B, N, Cp = tokens.shape
    assert N == h * w and Cp == 4 * C
    z = tokens.reshape(B, h, w, 2, 2, C)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    return z.reshape(B, C, 2 * h, 2 * w)
```

### 6.3 单元测试

必须实现：

```python
def test_pack_unpack_identity():
    z = torch.randn(2, 4, 64, 64)
    tokens = pack_2x2(z)
    z2 = unpack_2x2(tokens, h=32, w=32, C=4)
    assert torch.allclose(z, z2)
```

---

## 7. 几何边缘图模块

文件：`src/jcv2/models/geometry.py`

### 7.1 EdgeEnergyMap

```python
class EdgeEnergyMap(nn.Module):
    def __init__(self, kernel: str = "sobel", eps: float = 1e-6):
        super().__init__()
        # register_buffer sobel_x, sobel_y

    @torch.no_grad()
    def forward(self, depth: Tensor, token_hw: tuple[int, int]) -> Tensor:
        """
        depth: [B,1,H,W], range [0,1]
        token_hw: (h,w), e.g. (32,32)
        return: [B,N], range [0,1], detached
        """
```

实现步骤：

```text
1. clamp depth to [0,1]
2. conv2d with Sobel/Scharr kernels
3. grad_mag = sqrt(gx^2 + gy^2 + eps)
4. optional robust clip by percentile or fixed clamp
5. per-sample normalize to [0,1]
6. downsample to token grid h x w using area interpolation
7. flatten to [B,N]
8. detach
```

### 7.2 注意事项

- `E_loss` 必须从 pseudo-depth target 生成，且只用于 loss；
- `E_att` 在 depth-conditioned 任务中来自 input depth；
- joint generation 第一步 `E_att=0`；
- joint generation 后续步的 `E_att` 来自上一轮 depth clean estimate，不能来自当前 forward 的未来输出。

---

## 8. Controller 模块

文件：`src/jcv2/models/controller.py`

### 8.1 数据结构

```python
@dataclass
class TaskBatch:
    task: Literal["joint", "depth_conditioned"]
    t_x: Tensor        # [B]
    t_y: Tensor        # [B]
    m_x: Tensor        # [B]
    m_y: Tensor        # [B]
    lambda_x: Tensor   # [B]
    lambda_y: Tensor   # [B]
    task_id: Tensor    # [B], 0/1
```

### 8.2 时间采样

```python
class TimeSampler:
    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        # default uniform in [eps, 1-eps]
```

可选实现：

```text
uniform: t ~ U(eps, 1-eps)
logit_normal: sigmoid(N(mu, sigma))
flux_shifted: 根据 Flux timestep schedule 自定义
```

### 8.3 任务采样

```python
class TaskController(nn.Module):
    def sample_task_batch(self, batch_size: int, device: torch.device, train: bool) -> TaskBatch:
        # with probability p_joint_task sample joint else depth_conditioned
```

联合生成 timestep policy：

```python
if task == "joint":
    if torch.rand(()) < p_sync:
        t = sample_time(B)
        t_x = t_y = t
    else:
        t_x = sample_time(B)
        t_y = sample_time(B)
    m_x = m_y = 1
    lambda_x = lambda_y = 1
```

深度条件：

```python
t_x = sample_time(B)
t_y = zeros(B)
m_x = ones(B)
m_y = zeros(B)
lambda_x = ones(B)
lambda_y = zeros(B)
```

### 8.4 时间嵌入

```python
class BranchTimeEmbedding(nn.Module):
    def forward(self, t, task_id, m, lambda_) -> Tensor:
        # sinusoidal PE(t) + task/m/lambda embedding -> MLP -> hidden_dim
```

---

## 9. JointConn-v2 模块

文件：`src/jcv2/models/jointconn_v2.py`

### 9.1 输入输出

```python
@dataclass
class JointConnStats:
    gate: Tensor       # [B,1,1]
    r_x: Tensor        # [B,N,1]
    r_y: Tensor        # [B,N,1]
    r_0: Tensor        # [B,N,1]

class JointConnV2Block(nn.Module):
    def forward(
        self,
        h_x: Tensor,             # [B,N,d]
        h_y: Tensor,             # [B,N,d]
        e_att: Tensor,           # [B,N]
        token_hw: tuple[int,int],
        t_x: Tensor,             # [B]
        t_y: Tensor,             # [B]
        m_x: Tensor,             # [B]
        m_y: Tensor,             # [B]
        lambda_x: Tensor,        # [B]
        lambda_y: Tensor,        # [B]
        layer_index: int,
    ) -> tuple[Tensor, Tensor, JointConnStats]:
        """return residual_x, residual_y, stats"""
```

### 9.2 RelativePositionBias2D

文件：`src/jcv2/models/relative_position.py`

```python
class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int, max_h: int, max_w: int):
        self.table = nn.Parameter(torch.zeros(num_heads, 2*max_h-1, 2*max_w-1))
        # init truncated normal small std

    def forward(self, h: int, w: int) -> Tensor:
        """return [num_heads, N, N]"""
```

实现：

```text
每个 token 对应 h x w 网格坐标 (u_i, v_i)
du = u_i - u_j, dv = v_i - v_j
bias[head, i, j] = table[head, du + h - 1, dv + w - 1]
```

缓存 index：

```python
self._index_cache[(h,w,device)] = flat_indices
```

### 9.3 Pairwise Geometry Bias

修正后的几何 bias：

```text
B_ij^h = Pi^h(p_i - p_j) + beta_att * E_i * E_j * K_sigma(p_i, p_j)
K_sigma = exp(-||p_i-p_j||^2 / (2*sigma^2))
```

实现函数：

```python
def build_geometry_bias(e_att: Tensor, rel_pos_bias: Tensor, token_hw: tuple[int,int]) -> Tensor:
    """
    e_att: [B,N]
    rel_pos_bias: [H,N,N]
    return: [B,H,N,N]
    """
```

注意：

- 不要使用 `0.5*(E_i + E_j)` 作为 attention bias，因为 softmax 对每行常数平移不敏感，query 端常数项会被抵消；
- `E_i * E_j` 是真正 pairwise 的 edge-edge compatibility；
- local kernel 可以预先缓存 `[N,N]`。

### 9.4 Swap-Q Cross Attention

```python
class SwapQCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        self.q_x = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_x = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_x = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_x = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_y = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, h_x, h_y, bias):
        # bias: [B,H,N,N]
        # x <- y: softmax(Qx Ky^T / sqrt(dh) + bias) Vy
        # y <- x: softmax(Qy Kx^T / sqrt(dh) + bias) Vx
        return o_x, o_y
```

实现建议：

- 优先使用 `torch.nn.functional.scaled_dot_product_attention`；
- 若使用 additive bias，确保 mask shape 可广播为 `[B, heads, N, N]`；
- 对大 N=1024，注意显存；必要时提供 `use_memory_efficient_attention` 开关；
- 输出投影可 zero init，以保证插入模块初始时接近无影响。

### 9.5 Content Gate

```python
class ContentGate(nn.Module):
    def forward(self, h_x, h_y, t_x, t_y, m_x, m_y, lambda_x, lambda_y, layer_index):
        gap_x = h_x.detach().mean(dim=1)
        gap_y = h_y.detach().mean(dim=1)
        feat = concat([time_pe(t_x), time_pe(t_y), masks, lambdas, layer_embedding, gap_x, gap_y])
        gate = sigmoid(mlp(feat)).view(B, 1, 1)
        return gate
```

约束：

- `GAP(h_x)` 和 `GAP(h_y)` 必须 stop-gradient；
- gate 只调制 residual branch，不直接替代主干路径；
- gate 进入 loss weight 时必须再次 detach。

### 9.6 Regional Routing

使用三分类 routing：

```text
r_x: depth -> RGB injection strength
r_y: RGB -> depth injection strength
r_0: no-fusion state
r_x + r_y + r_0 = 1
```

```python
class RegionalRouting(nn.Module):
    def forward(self, h_x, h_y, e_att, t_x, t_y):
        # h_x,h_y: [B,N,d]
        # e_att: [B,N]
        # time features broadcast to [B,N,time_dim]
        logits = mlp(concat([h_x, h_y, e_att[...,None], time_x, time_y]))
        routing = softmax(logits, dim=-1)  # [B,N,3]
        r_x, r_y, r_0 = routing.split(1, dim=-1)
        return r_x, r_y, r_0
```

可选二路 sigmoid：

```text
r_x = sigmoid(logit_x)
r_y = sigmoid(logit_y)
no-fusion strength = 1 - max(r_x, r_y)  # only for logging
```

默认使用三分类 routing。

### 9.7 Layer-wise Coupling Schedule

```python
class LayerwiseCouplingSchedule(nn.Module):
    def forward(self, t_x, t_y, m_x, m_y, lambda_x, lambda_y):
        # returns w_x, w_y: [B,1,1]
        w = w_max * sigmoid(mlp([...]))
```

### 9.8 Residual Fusion

```python
res_x = lambda_x[:,None,None] * w_x * gate * r_x * o_x
res_y = lambda_y[:,None,None] * w_y * gate * r_y * o_y
```

在 depth-conditioned 任务中，`lambda_y=0`，因此 `res_y=0`。

---

## 10. Dual Flux / DiT 主干封装

文件：`src/jcv2/models/dual_flux.py`

### 10.1 总体类

```python
@dataclass
class DualFluxOutput:
    v_x: Tensor                 # [B,N,Cp]
    v_y: Tensor                 # [B,N,Cp]
    connector_stats: list[JointConnStats]

class JointConnDualFluxModel(nn.Module):
    def __init__(self, base_transformer, config):
        ...

    def forward(
        self,
        x_tokens: Tensor,       # [B,N,Cp]
        y_tokens: Tensor,       # [B,N,Cp]
        text: dict,
        task: TaskBatch,
        e_att: Tensor,          # [B,N]
        token_hw: tuple[int,int],
    ) -> DualFluxOutput:
        ...
```

### 10.2 推荐实现策略

不要直接在运行时用 fragile monkey-patch 修改 Diffusers forward。推荐：

1. 固定 diffusers 版本；
2. 将目标版本的 `FluxTransformer2DModel` 关键 forward/block 代码复制到本仓库，例如：

```text
src/jcv2/models/flux_patched.py
```

3. 在 block 循环中显式插入 JointConn-v2；
4. 通过单元测试确保当 `connector residual = 0` 时，单分支输出与原始 Flux forward 接近一致。

### 10.3 双分支共享权重

实现思路：

```text
同一个 frozen block/LoRA block 被分别应用到 RGB hidden states 和 Depth hidden states。
基础权重共享，LoRA 参数共享或按分支共享取决于配置。
默认：同一组 LoRA 参数作用于两个分支，保证参数效率。
JointConn-v2 独立管理跨分支通信。
```

伪代码：

```python
def forward(...):
    h_x = input_proj(x_tokens)
    h_y = input_proj(y_tokens)

    e_x = controller_time_embedding(task.t_x, task.task_id, task.m_x, task.lambda_x)
    e_y = controller_time_embedding(task.t_y, task.task_id, task.m_y, task.lambda_y)

    stats = []
    for layer_idx, block in enumerate(self.blocks):
        h_x = block(h_x, text_tokens=text["tokens"], pooled=text["pooled"], time_emb=e_x, img_ids=img_ids)
        h_y = block(h_y, text_tokens=text["tokens"], pooled=text["pooled"], time_emb=e_y, img_ids=img_ids)

        if layer_idx in self.connector_block_indices:
            res_x, res_y, st = self.connectors[layer_idx](
                h_x=h_x,
                h_y=h_y,
                e_att=e_att,
                token_hw=token_hw,
                t_x=task.t_x,
                t_y=task.t_y,
                m_x=task.m_x,
                m_y=task.m_y,
                lambda_x=task.lambda_x,
                lambda_y=task.lambda_y,
                layer_index=layer_idx,
            )
            h_x = h_x + res_x
            h_y = h_y + res_y
            stats.append(st)

    v_x = output_proj(h_x, time_emb=e_x)
    v_y = output_proj(h_y, time_emb=e_y)
    return DualFluxOutput(v_x=v_x, v_y=v_y, connector_stats=stats)
```

### 10.4 LoRA 注入

文件：`src/jcv2/models/lora_utils.py`

```python
def freeze_base_and_enable_lora(model, lora_config):
    for p in model.parameters():
        p.requires_grad_(False)
    # inject LoRA into target linear modules
    # ensure only LoRA params require grad
```

检查函数：

```python
def report_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total, trainable / total
```

必须保证：

```text
trainable = LoRA params + JointConn-v2 params
frozen = VAE + text encoders + base DiT weights
```

---

## 11. GCM-WFM Loss

文件：`src/jcv2/losses/gcm_wfm.py`

### 11.1 输入输出

```python
@dataclass
class GCMWFMLossOutput:
    loss: Tensor
    loss_x: Tensor
    loss_y: Tensor
    alpha_x: Tensor      # [B,N]
    alpha_y: Tensor      # [B,N]
    diagnostics: dict

class GCMWFMLoss(nn.Module):
    def forward(
        self,
        v_x: Tensor,             # [B,N,Cp]
        v_y: Tensor,             # [B,N,Cp]
        tau_x: Tensor,           # [B,N,Cp]
        tau_y: Tensor,           # [B,N,Cp]
        e_loss: Tensor,          # [B,N]
        task: TaskBatch,
        connector_stats: list[JointConnStats],
    ) -> GCMWFMLossOutput:
        ...
```

### 11.2 alpha 构造

```python
def time_weight(t):
    return 1.0 + gamma_t * t * (1.0 - t)
```

connector reliability：

```python
def aggregate_connector_q(stats, branch: Literal["x","y"]):
    if not stats or not use_connector_reliability_q:
        return torch.ones(B, N, device=device)

    gates = torch.stack([s.gate.squeeze(-1).squeeze(-1) for s in stats], dim=0)  # [L,B]
    if branch == "x":
        routes = torch.stack([s.r_x.squeeze(-1) for s in stats], dim=0)          # [L,B,N]
    else:
        routes = torch.stack([s.r_y.squeeze(-1) for s in stats], dim=0)

    gate_mean = gates.mean(dim=0)[:, None]        # [B,1]
    route_mean = routes.mean(dim=0)               # [B,N]
    q = q_eps + (gate_mean * route_mean).detach()
    q = q.clamp(q_min, q_max)
    return q
```

alpha：

```python
def build_alpha(t, e_loss, q):
    # t: [B], e_loss: [B,N], q: [B,N]
    alpha_tilde = time_weight(t)[:, None] * (1.0 + beta_loss * e_loss) * q
    denom = alpha_tilde.mean(dim=1, keepdim=True) + alpha_eps
    alpha = alpha_tilde / denom
    alpha = alpha.clamp(alpha_min, alpha_max)
    return alpha.detach()
```

### 11.3 Loss 计算

```python
err_x = (v_x - tau_x).pow(2).mean(dim=-1)     # [B,N]
err_y = (v_y - tau_y).pow(2).mean(dim=-1)     # [B,N]
loss_x_per_sample = (alpha_x * err_x).mean(dim=1)
loss_y_per_sample = (alpha_y * err_y).mean(dim=1)

loss_x = (task.m_x * loss_x_per_sample).sum() / (task.m_x.sum() + eps)
loss_y = (task.m_y * loss_y_per_sample).sum() / (task.m_y.sum() + eps)
loss = loss_x + loss_y
```

若某 batch 内没有 joint sample，`task.m_y.sum()==0`，则 `loss_y=0`。

### 11.4 diagnostics

必须记录：

```text
alpha_x mean/std/min/max
alpha_y mean/std/min/max
ESS/N = (sum alpha)^2 / (N * sum alpha^2)
gate mean
routing r_x/r_y/r_0 mean
routing entropy
time mean
loss_x/loss_y
```

---

## 12. 训练流程

文件：`src/jcv2/training/trainer.py`

### 12.1 训练单步伪代码

```python
def training_step(batch):
    image = batch["image"].to(device, dtype)
    depth_rgb = batch["depth_rgb"].to(device, dtype)
    depth_1ch = batch["depth"].to(device, dtype)
    captions = batch["caption"]

    with torch.no_grad():
        z_x_data = vae.encode(image)          # [B,C,H',W']
        z_y_data = vae.encode(depth_rgb)      # [B,C,H',W']
        text = text_encoders.encode(captions)

    B, C, H_lat, W_lat = z_x_data.shape
    h, w = H_lat // 2, W_lat // 2

    task = controller.sample_task_batch(B, device=device, train=True)

    eps_x = torch.randn_like(z_x_data)
    eps_y = torch.randn_like(z_y_data)

    t_x = task.t_x.view(B, 1, 1, 1)
    t_y = task.t_y.view(B, 1, 1, 1)

    z_x_t = (1 - t_x) * z_x_data + t_x * eps_x
    z_y_t = (1 - t_y) * z_y_data + t_y * eps_y

    # depth-conditioned: t_y=0 makes z_y_t clean automatically
    tau_x = eps_x - z_x_data
    tau_y = eps_y - z_y_data

    x_tokens = pack_2x2(z_x_t)
    y_tokens = pack_2x2(z_y_t)
    tau_x_tokens = pack_2x2(tau_x)
    tau_y_tokens = pack_2x2(tau_y)

    e_loss = edge_energy(depth_1ch, token_hw=(h, w))

    if task is entirely depth_conditioned:
        e_att = edge_energy(depth_1ch, token_hw=(h, w))
    else:
        if cfg.train.joint_train_e_att_mode == "zero":
            e_att = torch.zeros_like(e_loss)
        elif cfg.train.joint_train_e_att_mode == "self_condition_no_grad":
            e_att = compute_self_conditioned_e_att_no_grad(...)

    out = model(
        x_tokens=x_tokens,
        y_tokens=y_tokens,
        text=text,
        task=task,
        e_att=e_att,
        token_hw=(h, w),
    )

    loss_out = gcm_wfm_loss(
        v_x=out.v_x,
        v_y=out.v_y,
        tau_x=tau_x_tokens,
        tau_y=tau_y_tokens,
        e_loss=e_loss,
        task=task,
        connector_stats=out.connector_stats,
    )

    return loss_out
```

注意：若一个 batch 内混合 joint 与 depth-conditioned 样本，则 `e_att` 需要按样本构造：

```python
e_att = torch.where(task_is_depth_conditioned[:,None], edge_depth, zero_or_self_condition)
```

### 12.2 Trainer 主循环

使用 Accelerate：

```python
accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
model, optimizer, dataloader, scheduler = accelerator.prepare(...)

for step, batch in enumerate(dataloader):
    with accelerator.accumulate(model):
        loss_out = training_step(batch)
        accelerator.backward(loss_out.loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainable_params, cfg.train.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

### 12.3 优化器

```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=2e-4,
    betas=(0.9, 0.95),
    weight_decay=0.05,
)
```

### 12.4 Checkpoint 保存

保存内容：

```text
output_dir/checkpoint-STEP/
  jointconn_v2.safetensors
  lora_adapter/
    adapter_config.json
    adapter_model.safetensors
  optimizer.pt
  scheduler.pt
  trainer_state.json
  config.yaml
```

不要保存冻结主干权重，除非显式配置 `save_full_model=true`。

---

## 13. 推理：Depth-conditioned RGB Sampling

文件：`src/jcv2/inference/samplers.py`

### 13.1 Pipeline 接口

```python
class DepthConditionedPipeline:
    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        depth: PIL.Image | Tensor,
        num_steps: int = 40,
        cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
    ) -> dict:
        return {
            "images": list[PIL.Image],
            "depth_condition": Tensor,
        }
```

### 13.2 Heun 采样流程

```python
# encode text
text_cond = text_encoders.encode(prompts)
text_uncond = text_encoders.encode([""] * B)

# encode clean depth
D = preprocess_depth(depth)                  # [B,1,H,W], [0,1]
D_rgb = depth_to_rgb_vae_input(D)            # [B,3,H,W]
z_y = vae.encode(D_rgb)                      # clean, fixed
E_att = edge_energy(D, token_hw=(h,w))        # fixed

y_tokens = pack_2x2(z_y)

# initialize RGB noise
z_x = torch.randn(B, C, H_lat, W_lat)

time_grid = make_descending_time_grid(num_steps)  # [1 ... 0]

for s in range(num_steps):
    t = time_grid[s]
    t_next = time_grid[s+1]
    dt = t - t_next

    task = make_depth_conditioned_task(B, t_x=t, t_y=0)

    x_tokens = pack_2x2(z_x)

    out_cond = model(x_tokens, y_tokens, text_cond, task, E_att, token_hw)
    out_uncond = model(x_tokens, y_tokens, text_uncond, task, E_att, token_hw)

    v_x = out_uncond.v_x + cfg_scale * (out_cond.v_x - out_uncond.v_x)
    v_x_latent = unpack_2x2(v_x, h, w, C)

    # Heun predictor
    z_tilde = z_x - dt * v_x_latent

    # second evaluation
    task_next = make_depth_conditioned_task(B, t_x=t_next, t_y=0)
    x_tilde_tokens = pack_2x2(z_tilde)

    out_cond_2 = model(x_tilde_tokens, y_tokens, text_cond, task_next, E_att, token_hw)
    out_uncond_2 = model(x_tilde_tokens, y_tokens, text_uncond, task_next, E_att, token_hw)
    v_x_2 = out_uncond_2.v_x + cfg_scale * (out_cond_2.v_x - out_uncond_2.v_x)
    v_x_2_latent = unpack_2x2(v_x_2, h, w, C)

    z_x = z_x - 0.5 * dt * (v_x_latent + v_x_2_latent)

image = vae.decode_rgb(z_x)
```

约束：

```text
- y/depth latent 全程不更新；
- lambda_y=0；
- m_y=0；
- CFG 只 drop text，不 drop depth condition；
- E_att 全程固定。
```

---

## 14. 推理：Joint RGB-Depth Sampling

文件：`src/jcv2/inference/samplers.py`

### 14.1 Pipeline 接口

```python
class JointGenerationPipeline:
    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        num_steps: int = 40,
        cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
    ) -> dict:
        return {
            "images": list[PIL.Image],
            "depths": list[PIL.Image],
        }
```

### 14.2 Euler / Heun 流程

```python
z_x = torch.randn(B, C, H_lat, W_lat)
z_y = torch.randn(B, C, H_lat, W_lat)
E_att = torch.zeros(B, h*w, device=device)

for s in range(num_steps):
    t = time_grid[s]
    t_next = time_grid[s+1]
    dt = t - t_next

    task = make_joint_task(B, t_x=t, t_y=t)

    x_tokens = pack_2x2(z_x)
    y_tokens = pack_2x2(z_y)

    out_cond = model(x_tokens, y_tokens, text_cond, task, E_att, token_hw)
    out_uncond = model(x_tokens, y_tokens, text_uncond, task, E_att, token_hw)

    v_x = out_uncond.v_x + cfg_scale * (out_cond.v_x - out_uncond.v_x)
    v_y = out_uncond.v_y + depth_cfg_scale * (out_cond.v_y - out_uncond.v_y)

    v_x_lat = unpack_2x2(v_x, h, w, C)
    v_y_lat = unpack_2x2(v_y, h, w, C)

    z_x = z_x - dt * v_x_lat
    z_y = z_y - dt * v_y_lat

    # Causal edge update for next step.
    if cfg.infer.joint_e_att_update == "causal_depth" and (s + 1) % update_e_att_every == 0:
        # clean estimate under current time t
        z0_y_est = z_y - t_next * v_y_lat
        D_est = vae.decode_depth(z0_y_est)
        E_att = edge_energy(D_est, token_hw=(h,w))

image = vae.decode_rgb(z_x)
depth = vae.decode_depth(z_y)
```

可用 Heun 替换 Euler；Heun 下 causal edge update 使用 corrector 后的 `z_y` 和第二次预测 `v_y_2` 更合理。

### 14.3 注意事项

- `E_att` 第一步必须为 0，除非外部给定 depth condition；
- 不能在当前 forward 前使用当前 forward 的输出构造 `E_att`；
- 对 joint generation，可选择每 `k` 步更新一次 `E_att` 以节省 decode 成本；
- depth decode 只用于构造边缘图时应 `torch.no_grad()`。

---

## 15. Evaluation Metrics

文件：`src/jcv2/metrics/`

### 15.1 Depth scale-shift normalization

对预测深度 `D_hat` 做 per-sample scale-shift：

```text
D_tilde = a * D_hat + b
使 median(D_tilde)=0.5, p90(D_tilde)=0.95
```

实现：

```python
def scale_shift_normalize_depth(d: Tensor) -> Tensor:
    med = quantile(d, 0.5)
    p90 = quantile(d, 0.9)
    a = (0.95 - 0.5) / (p90 - med + eps)
    b = 0.5 - a * med
    return (a * d + b).clamp(0, 1)
```

### 15.2 Edge-F1

按照论文附录 D：

```text
Image edges:
  - RGB -> grayscale
  - gamma correction gamma=2.2
  - Canny thresholds T1=0.1, T2=0.3
  - NMS
  - 1px dilation

Depth edges:
  - scale-shift normalized depth
  - Sobel gradients
  - magnitude + NMS + threshold
  - 1px dilation

Precision = |E_I ∩ dilate(E_D,1)| / |E_I|
Recall    = |E_D ∩ dilate(E_I,1)| / |E_D|
F1        = 2PR/(P+R)
```

### 15.3 GradCorr / Grad-Cos

```python
def grad_cosine(image_gray, depth):
    grad_i = sobel(image_gray)
    grad_d = sobel(depth)
    cos = dot(grad_i, grad_d) / (norm(grad_i) * norm(grad_d) + eps)
    return cos.mean(), (angle < 15deg).float().mean()
```

### 15.4 Depth-conditioned adherence metrics

```text
1. Generate RGB image from (T,D)
2. Re-estimate depth D_re from generated image using same depth estimator
3. scale-shift normalize D_re
4. compute RMSE(D_re, D), SSIM(D_re, D), GradCorr, Edge-F1
```

### 15.5 Joint generation metrics

```text
FID
CLIPScore
RMSE between generated depth and paired pseudo-depth after scale alignment
GradCorr
Cycle-RMSE if implemented
Edge-F1
```

---

## 16. Logging 与可视化

### 16.1 训练日志

每 `log_every_steps` 记录：

```text
loss_total
loss_x
loss_y
alpha_x_mean/std/min/max
alpha_y_mean/std/min/max
alpha_ess_x
alpha_ess_y
gate_mean
r_x_mean
r_y_mean
r_0_mean
routing_entropy
lr
grad_norm
```

### 16.2 可视化样例

保存：

```text
RGB input / generated RGB
Depth target / generated depth
E_loss heatmap
E_att heatmap
alpha heatmap
routing r_x/r_y/r_0 heatmap
gate scalar distribution
```

---

## 17. 测试清单

### 17.1 packing 测试

```text
pack + unpack = identity
norm preserving
shape correct under C=4 and C=16
```

### 17.2 geometry 测试

```text
constant depth -> E approx zero
single vertical edge -> E high near edge
output shape [B,N]
range [0,1]
detached: requires_grad=False
```

### 17.3 controller 测试

```text
joint task: m_x=m_y=1, lambda_x=lambda_y=1
depth-conditioned: m_x=1, m_y=0, lambda_x=1, lambda_y=0, t_y=0
p_sync roughly works
```

### 17.4 JointConn 测试

```text
input h_x,h_y [B,N,d] -> residuals [B,N,d]
routing sums to 1
r_0 exists
lambda_y=0 -> residual_y exactly zero
zero-init connector -> residual norm near zero
attention bias shape [B,H,N,N]
```

### 17.5 GCM-WFM 测试

```text
alpha detached
alpha bounded in [alpha_min, alpha_max]
per-sample mean approx 1 before clipping, bounded after clipping
m_y=0 -> loss_y = 0
if alpha=1 and masks both 1 -> equivalent to MSE over packed velocity
```

### 17.6 inference 测试

```text
depth-conditioned sampling does not modify y latent
joint sampling produces both image and depth
CFG drop text only; depth condition remains fixed
```

---

## 18. Codex 实现顺序

建议让 Codex 按以下顺序实现，不要一次性生成全部代码。

### Phase 1: 基础工具

1. `config.py`
2. `utils/seed.py`, `utils/tensor.py`, `utils/checkpoint.py`
3. `models/packing.py`
4. `models/geometry.py`
5. 单元测试：packing + geometry

### Phase 2: 数据

1. `data/transforms.py`
2. `data/coco_dataset.py`
3. `data/collate.py`
4. `scripts/prepare_coco_depth.py`

### Phase 3: Controller + Loss

1. `models/controller.py`
2. `losses/gcm_wfm.py`
3. 单元测试：controller + loss

### Phase 4: JointConn-v2

1. `models/relative_position.py`
2. `models/jointconn_v2.py`
3. 单元测试：routing、gate、residual、bias shape

### Phase 5: Backbone wrapper

1. `models/autoencoder.py`
2. `models/text_encoder.py`
3. `models/lora_utils.py`
4. `models/dual_flux.py`
5. smoke test：随机 tokens forward

### Phase 6: Training

1. `training/optim.py`
2. `training/train_state.py`
3. `training/trainer.py`
4. `scripts/train.py`
5. one-batch overfit test

### Phase 7: Inference

1. `inference/schedulers.py`
2. `inference/cfg.py`
3. `inference/samplers.py`
4. `inference/pipelines.py`
5. `scripts/infer_joint.py`
6. `scripts/infer_depth_conditioned.py`

### Phase 8: Evaluation

1. `metrics/edge_metrics.py`
2. `metrics/depth_metrics.py`
3. `metrics/fid_clip.py`
4. `scripts/eval_depth_edges.py`

---

## 19. 关键实现陷阱

### 19.1 不要训练冻结组件

训练前后检查：

```python
for name, p in model.named_parameters():
    if p.requires_grad:
        assert "lora" in name.lower() or "jointconn" in name.lower() or "connector" in name.lower()
```

### 19.2 不要让 alpha 产生梯度

```python
alpha = alpha.detach()
```

并在测试中检查：

```python
assert not alpha.requires_grad
```

### 19.3 不要在 depth-conditioned 中更新 depth branch

必须同时保证：

```text
m_y = 0
lambda_y = 0
t_y = 0
z_y fixed during sampler loop
```

### 19.4 不要泄漏 clean depth 到 joint attention

联合生成训练：

```text
E_loss 可用 target depth；
E_att 不可用 target depth。
```

如果实现 `self_condition_no_grad`，它必须来自当前模型预测，并且 no-grad。

### 19.5 不要硬编码 latent channel

```python
C = z_x_data.shape[1]
Cp = 4 * C
```

### 19.6 注意 Flux 内部 API

Diffusers 的 Flux 内部 forward 可能变更。推荐固定 diffusers 版本并复制 patched transformer。不要依赖不稳定的私有属性名，除非有单元测试覆盖。

---

## 20. 最小配置样例

文件：`configs/train_jointconn_v2.yaml`

```yaml
data:
  dataset_name: coco2017
  image_root: data/coco/train2017
  caption_file: data/coco/annotations/captions_train2017.json
  depth_root: data/coco_depth_v2/train2017
  resolution: 512
  center_crop: true
  random_flip: true

model:
  backbone_type: flux
  pretrained_model_name_or_path: black-forest-labs/FLUX.1-dev
  mode: hf_flux_mode
  latent_channels: null
  pack_size: 2
  connector_block_indices: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
  connector_in_single_stream: true
  freeze_vae: true
  freeze_text_encoders: true
  freeze_base_transformer: true
  use_lora: true
  lora_rank: 16
  lora_alpha: 16
  lora_dropout: 0.0
  lora_target_modules: [to_q, to_k, to_v, to_out, ff, proj]

jointconn:
  beta_att: 1.0
  local_kernel_sigma: 3.0
  use_pairwise_edge_bias: true
  routing_type: three_way
  use_no_fusion_state: true
  gate_hidden_dim: 512
  routing_hidden_dim: 256
  residual_w_max: 1.0
  zero_init_output_proj: true
  dropout: 0.0

loss:
  beta_loss: 2.0
  gamma_t: 1.0
  alpha_min: 0.25
  alpha_max: 4.0
  alpha_eps: 1.0e-6
  use_connector_reliability_q: true
  q_eps: 0.05
  q_min: 0.05
  q_max: 2.0
  normalize_alpha_per_sample: true
  detach_alpha: true

train:
  output_dir: outputs/jointconn_v2
  seed: 42
  train_batch_size: 4
  gradient_accumulation_steps: 1
  mixed_precision: bf16
  max_train_steps: 100000
  lr: 2.0e-4
  adam_beta1: 0.9
  adam_beta2: 0.95
  weight_decay: 0.05
  max_grad_norm: 1.0
  gradient_checkpointing: true
  p_joint_task: 0.5
  p_sync: 0.5
  time_sampling: uniform
  joint_train_e_att_mode: zero
  save_every_steps: 2000
  log_every_steps: 50
  num_workers: 8
```

---

## 21. 命令行接口设计

### 21.1 训练

```bash
accelerate launch scripts/train.py \
  --config configs/train_jointconn_v2.yaml
```

### 21.2 深度条件生成

```bash
python scripts/infer_depth_conditioned.py \
  --checkpoint outputs/jointconn_v2/checkpoint-100000 \
  --prompt "A white robot standing in a bright modern kitchen." \
  --depth path/to/depth.png \
  --out outputs/samples/depth_conditioned.png \
  --num-steps 40 \
  --cfg-scale 4.0
```

### 21.3 联合生成

```bash
python scripts/infer_joint.py \
  --checkpoint outputs/jointconn_v2/checkpoint-100000 \
  --prompt "A calm ocean with waves gently lapping at the shore." \
  --out-dir outputs/samples/joint \
  --num-steps 40 \
  --cfg-scale 4.0
```

### 21.4 评估

```bash
python scripts/eval_depth_edges.py \
  --pred-root outputs/eval/pred \
  --depth-root data/coco_depth_v2/val2017 \
  --caption-file data/coco/annotations/captions_val2017.json
```

---

## 22. 参考资料与实现依据

论文中的实现依据：

```text
- 双分支 Flux-style DiT，MM-DiT -> P-DiT；
- 支持 joint generation 和 depth-conditioned generation；
- 冻结主干，仅训练 LoRA 与 JointConn-v2；
- 2x2 packing，512x512 图像，latent grid 64x64，packed token grid 32x32；
- COCO 2017，Depth Anything V2 pseudo-depth；
- AdamW β1=0.9, β2=0.95, weight_decay=0.05, lr=2e-4；
- 50% synchronous / 50% independent timestep policy；
- GCM-WFM 使用 temporal、edge、gate、routing 的对角权重，stop-gradient，per-sample normalization；
- depth-conditioned inference 使用 40 steps + Heun。
```

本实现对 Method 做了以下必要修正：

```text
1. 统一 reverse-time flow sign convention；
2. 引入 branch masks m_x/m_y 和 coupling masks lambda_x/lambda_y；
3. 将 E_att 与 E_loss 分离，避免 joint generation target leakage；
4. 将 routing 从二方向 softmax 改为三分类 routing，显式加入 no-fusion state；
5. 将 attention edge bias 改为 pairwise E_i E_j local compatibility；
6. 将 alpha 定义为 detached normalized clipped bounded preconditioner。
```

外部工程参考：

```text
- Hugging Face Diffusers FluxTransformer2DModel:
  https://huggingface.co/docs/diffusers/api/models/flux_transformer

- Hugging Face Diffusers Flux pipeline / Flux control examples:
  https://huggingface.co/docs/diffusers/api/pipelines/flux

- Diffusers LoRA loading and adapter utilities:
  https://huggingface.co/docs/diffusers/api/loaders/lora

- PEFT LoRA documentation:
  https://huggingface.co/docs/peft/package_reference/lora

- Depth Anything V2 Transformers documentation:
  https://huggingface.co/docs/transformers/en/model_doc/depth_anything_v2

- Depth Anything V2 official repository:
  https://github.com/DepthAnything/Depth-Anything-V2
```

---

## 23. Definition of Done

代码实现完成后，必须满足：

```text
[ ] 单测全部通过；
[ ] one-batch overfit 能让 loss 下降；
[ ] depth-conditioned inference 中 depth latent 不变；
[ ] joint inference 能输出 RGB 和 depth；
[ ] checkpoint 只包含 LoRA + JointConn-v2；
[ ] alpha 始终 detached 且 bounded；
[ ] E_att/E_loss 逻辑清晰，无 target leakage；
[ ] config 可切换 paper_mode 与 hf_flux_mode；
[ ] 支持 512x512 默认训练；
[ ] 支持 40-step Heun sampling；
[ ] 评估脚本能输出 FID、CLIPScore、RMSE、SSIM、GradCorr、Edge-F1。
```

---

## 24. 当前 JointConn-v2 工程适配实现方案

> 本节是面向当前 `D:\code\JointConn-v2` 代码库的落地修正版。前文 `src/jcv2/...` 目录结构适合新建工程；当前实现应优先复用已有 `jointconn_v2_library + train.py + inference.py` 主链路，避免重写 Flux / sd-scripts 训练框架。

### 24.1 当前仓库中的关键事实

当前项目已经具备以下 JointConn-v2 主链路结构：

```text
jointconn_v2_library/jointconn_v2_model.py
  - JointConn-v2 主模型
  - DoubleStreamBlock / SingleStreamBlock
  - 当前 joint_attention
  - 当前 joint1 / joint2 residual adapter

jointconn_v2_library/jointconn_v2_utils.py
  - load_empty_flux_model
  - setup_jointconn_v2_model
  - save_added_params
  - timestep sampling helpers

jointconn_v2_library/inference_pipeline.py
  - joint_generation
  - conditional_generation
  - denoise / conditional_denoise

train.py
  - 主训练入口
  - RGB/depth latent 拼 batch 训练
  - 当前 flow matching loss

inference.py
  - 主推理入口
  - 加载 Flux base + JointConn-v2 addons
```

当前模型不是两个完全独立的 `h_x/h_y` module forward，而是将 RGB 和 Depth 作为 batch 维度的前后两半：

```text
batch[:B]  = RGB branch
batch[B:]  = Depth branch
```

因此 JointConn-v2 在当前仓库中应实现为 **batch-split cross-connector**，替换或升级现有 `joint_attention + joint1/joint2`，而不是新增一个完整 `DualFluxModel` 重构主干。

### 24.2 维度约定修正

论文叙述中的示例常写：

```text
latent_channels = 4
packed_channels = 16
```

但当前 Flux-style JointConn-v2 代码配置为：

```text
latent_channels = 16
packed_channels = 64
JointConnV2Model.params.in_channels = 64
```

实现中不得硬编码 `C=4` 或 `packed_channels=16`。所有 token 维度必须从实际 tensor 或模型配置读取：

```python
packed_channels = img.shape[-1]
latent_channels = packed_channels // 4
```

文档中的 `paper_mode / hf_flux_mode` 在当前仓库中应改写为：

```text
paper_formula_mode:
  仅用于论文公式解释，可能使用 C=4 的示例。

current_jointdit_mode:
  当前代码实际模式，VAE latent C=16，packed token dim=64。
```

### 24.3 推荐新增文件

不要新建完整 `src/jcv2` 工程。建议新增以下文件：

```text
jointconn_v2_library/geometry.py
  EdgeEnergyMap
  build_zero_e_att
  build_depth_e_att

jointconn_v2_library/relative_position.py
  RelativePositionBias2D
  local Gaussian kernel cache

jointconn_v2_library/jointconn_v2.py
  JointConnV2Block
  JointConnStats
  SwapQCrossAttention
  ContentGate
  RegionalRouting
  LayerwiseCouplingSchedule

jointconn_v2_library/gcm_wfm.py
  TaskBatch
  GCMWFMLoss
  alpha diagnostics

jointconn_v2_library/controller.py
  可选：TaskController / timestep policy
```

后续如果需要最小单测，可新增：

```text
tests/test_jointconn_v2.py
tests/test_gcm_wfm.py
tests/test_geometry.py
```

如果当前环境没有测试框架，也可以先提供 `scripts/smoke_jointconn_v2.py` 做轻量 shape / gradient 检查。

### 24.4 需要修改的现有文件

#### 24.4.1 `jointconn_v2_library/jointconn_v2_utils.py`

`setup_jointconn_v2_model` 当前会加入 `joint1/joint2` 和 LoRA。需要改为可配置：

```text
--jointconn_version v1 | v2
--enable_jointconn_v2
--jointconn_beta_att
--jointconn_routing_type three_way | two_sigmoid
--jointconn_use_edge_bias
--jointconn_use_rel_pos_bias
--jointconn_zero_init
--jointconn_lambda_y
```

实现原则：

```text
v1: 保持原始 joint1/joint2 逻辑，用于 baseline 和兼容旧 checkpoint。
v2: 给每个 double/single block 挂载 jointconn_v2 模块。
```

保存权重时，`save_added_params` 必须包含：

```text
lora_A / lora_B
joint1 / joint2   # v1 兼容
jointconn_v2      # v2 新模块
rel_pos_bias
router
content_gate
coupling_schedule
```

#### 24.4.2 `jointconn_v2_library/jointconn_v2_model.py`

需要扩展 `JointConnV2Model.forward` 参数：

```python
def forward(
    ...,
    e_att: Optional[Tensor] = None,          # [B,N] where B is pair batch size before RGB/depth duplication
    task_masks: Optional[dict] = None,       # m_x, m_y, lambda_x, lambda_y
    return_jointconn_stats: bool = False,
):
```

当前 `timesteps` 是 `[2B]`，其中前半是 RGB，后半是 Depth。JointConn-v2 内部应拆分：

```python
t_x = timesteps[:B]
t_y = timesteps[B:]
```

若 `task_masks is None`，默认：

```text
m_x = 1
m_y = 1
lambda_x = 1
lambda_y = 1
```

若 `e_att is None`，默认使用全零 edge map，以保证旧推理路径可运行。

DoubleStreamBlock 中：

```text
img token 和 text token 是分离的。
JointConn-v2 只作用于 img hidden tokens。
```

SingleStreamBlock 中：

```text
x = concat(text, image)
必须用 txt_len 切分：
  txt_part = x[:, :txt_len]
  img_part = x[:, txt_len:]
JointConn-v2 只能作用 img_part。
处理完再 concat 回去。
```

这点非常重要，因为 `E_att/routing/alpha` 只对应 spatial image token grid，不对应 text tokens。

#### 24.4.3 `train.py`

当前训练逻辑只接近 joint generation 训练，需要加入任务采样：

```text
p_joint_task:
  采 joint RGB-depth generation。

1 - p_joint_task:
  采 depth-conditioned image generation。
```

joint task：

```text
m_x = 1, m_y = 1
lambda_x = 1, lambda_y = 1
50% t_x=t_y
50% t_x,t_y independent
```

depth-conditioned task：

```text
m_x = 1, m_y = 0
lambda_x = 1, lambda_y = 0
t_x sampled
t_y = 0
z_y_t = clean depth latent
loss only on RGB branch
```

当前原代码中 “small timestep + normal timestep” 的不平衡采样可以保留为 v1 baseline，但 v2 默认应切换为文档中的：

```text
p_sync = 0.5
p_independent = 0.5
```

GCM-WFM loss 建议在 packed token space 计算：

```text
model_pred:       [2B,N,Cp]
target velocity:  [2B,N,Cp]
alpha_x/y:        [B,N]
```

流程为：

```text
1. 构造 z_x_t, z_y_t。
2. pack noisy latents。
3. pack tau_x/tau_y = eps - data。
4. forward 得到 packed velocity。
5. split pred into v_x/v_y。
6. 用 GCMWFMLoss 计算 branch-masked token loss。
```

不要先 unpack 成 `[B,C,H,W]` 再做普通 image-space MSE，否则 token-wise alpha 会变得别扭且不符合文档。

#### 24.4.4 `jointconn_v2_library/inference_pipeline.py`

需要给 joint generation 和 conditional generation 增加 v2 路径。

Joint generation：

```text
E_att 初始为 0。
如果 joint_e_att_update=causal_depth，则每 k 步：
  1. 根据当前 depth latent 和当前 v_y 估计 clean depth。
  2. decode depth。
  3. EdgeEnergyMap -> E_att。
下一步 forward 使用这个 E_att。
```

Depth-conditioned image generation：

```text
输入 depth map -> E_att 固定。
lambda_y=0。
m_y=0。
depth latent 全程固定或每 step 重置为 condition latent。
```

当前 `conditional_denoise` 已经通过替换 batch 中某一半 latent 来固定条件分支，可在此基础上接入：

```text
gen_type == depth_to_image:
  condition 是 depth branch，lambda_y=0，m_y=0，E_att=Edge(input_depth)

gen_type == depth_estimation:
  如果继续支持 RGB->Depth，则可对称设置 lambda_x=0, m_x=0。
  但 method_intro 当前主任务是 depth-conditioned image generation，RGB->Depth 可作为兼容任务而非 v2 主实验。
```

### 24.5 LoRA 策略需要统一

当前 `LoRALinear` 的 forward 是：

```text
前半 batch: base layer
后半 batch: base layer + LoRA
```

这意味着当前 LoRA 主要作用在 Depth branch。JointConn-v2 文档默认假设 LoRA 作用于两分支共享路径。两者不一致，必须显式选择。

推荐 v2 实现采用可配置策略：

```text
lora_branch_mode = depth_only | shared_both | separate
```

初始默认建议：

```text
depth_only:
  保持旧 checkpoint / baseline 兼容，改动最小。

shared_both:
  更符合 JointConn-v2 方法表述，但会改变旧 baseline 行为。
```

编码阶段如果目标是先复现丢失代码并快速跑通，建议第一版保留 `depth_only`，后续消融中再加入 `shared_both`。

### 24.6 E_att / E_loss 在当前训练中的构造

当前数据集中 depth 有两种形式：

```text
latent training:
  depth_latent 已预缓存，但通常没有 1-channel depth tensor。

non-latent training:
  batch["depth"] 是 3-channel depth image tensor，range [-1,1]。
```

GCM-WFM 需要 `E_loss [B,N]`。因此：

```text
non-latent training:
  depth_1ch = mean((depth + 1) / 2, channel)
  E_loss = EdgeEnergyMap(depth_1ch)

latent training:
  推荐新增 depth edge cache：
    image_folder/depth_edges/{fname}.pt
  或在 preprocess_step2 阶段同时保存 E_loss。
```

第一版编码建议：

```text
优先支持 non-latent 或增加 depth_edges cache。
如果只跑 latent training，则必须先实现 edge cache，否则 loss 只能退化为 E_loss=0。
```

为了避免 target leakage：

```text
joint task:
  E_loss 可用 target depth edge。
  E_att 默认 0，不用 target depth edge。

depth-conditioned task:
  E_loss 可用 input/target depth edge。
  E_att 使用 input depth edge。
```

### 24.7 配置项映射到当前 CLI

当前项目主要靠 argparse。建议先在 `train.py` 和 `inference.py` 增加 argparse 参数，而不是立即引入 YAML config：

```text
--enable_jointconn_v2
--jointconn_beta_att 1.0
--jointconn_local_kernel_sigma 3.0
--jointconn_routing_type three_way
--jointconn_alpha_min 0.25
--jointconn_alpha_max 4.0
--jointconn_beta_loss 2.0
--jointconn_use_reliability_q
--p_joint_task 0.5
--p_sync 0.5
--joint_train_e_att_mode zero
--joint_e_att_update zero | causal_depth
--joint_e_att_update_every 1
--lora_branch_mode depth_only | shared_both | separate
```

保存 adapter JSON 时也要记录这些 flags，供推理自动恢复。

### 24.8 最小实现顺序

当前仓库推荐按以下顺序执行：

```text
Phase A: 文档和兼容开关
  1. 增加 argparse flags。
  2. 默认 enable_jointconn_v2=False，保证旧代码可跑。

Phase B: 几何和 loss 基础
  1. jointconn_v2_library/geometry.py
  2. jointconn_v2_library/gcm_wfm.py
  3. smoke tests for shape / detach / alpha bounds。

Phase C: JointConn-v2 模块
  1. jointconn_v2_library/relative_position.py
  2. jointconn_v2_library/jointconn_v2.py
  3. 替换 DoubleStreamBlock / SingleStreamBlock 中旧 joint residual。
  4. zero-init 下确认输出接近旧模型。

Phase D: 训练接入
  1. train.py 构造 TaskBatch。
  2. E_loss/E_att 接入。
  3. packed-space GCM-WFM loss。
  4. one-batch smoke train。

Phase E: 推理接入
  1. depth_to_image 固定 E_att。
  2. joint_generation E_att=0。
  3. 可选 causal_depth update。
```

### 24.9 当前文档与方法简介的一致性修正

`method_intro.md` 中的方法逻辑保持有效，但代码设计文档前文若继续保留 `src/jcv2` 新工程结构，需要明确：

```text
前文是理想新工程结构；
第 24 节是当前 JointConn-v2 工程实际编码准则；
实际编码以第 24 节为准。
```

尤其以下内容以第 24 节覆盖前文：

```text
1. 不新建完整 src/jcv2 工程。
2. 不复制 diffusers FluxTransformer2DModel。
3. 不重写 DualFlux wrapper。
4. 使用当前 JointConn-v2 batch-split RGB/depth 分支。
5. packed token dim 使用当前模型实际 64，而不是论文示例 16。
6. JointConn-v2 只作用 spatial image tokens，不作用 text tokens。
```

### 24.10 编码前需要用户确认的两个选择

正式实现前建议确认：

```text
1. LoRA branch mode:
   A. depth_only：保持原 JointDiT 兼容，第一版推荐。
   B. shared_both：更贴合新方法，但 baseline 行为会变。

2. v2 第一版是否覆盖 RGB->Depth depth_estimation:
   A. 只实现论文主线 joint_generation + depth_to_image。
   B. 同时保留并适配 depth_estimation。
```

如果没有额外说明，默认执行：

```text
lora_branch_mode = depth_only
v2 tasks = joint_generation + depth_to_image
depth_estimation = 保留旧逻辑兼容
```
