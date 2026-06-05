# JointConn-v2 with GCM-WFM：3 分钟无语音方法框架视频设计 Prompt

> 用途：将本文方法实现框架制作成约 3 分钟学术介绍视频。视频不需要语音，必须使用完整字幕，字幕内容与画面严格对应。本文档可直接交给视频生成模型、动画设计工具或视频制作人员使用。

---

## 1. 视频总体设定

**视频主题**：JointConn-v2 with GCM-WFM：用于 RGB–Depth 联合生成与深度条件图像生成的几何感知双分支 Diffusion Transformer 框架。  
**目标时长**：180 秒，允许误差 ±3 秒。  
**画幅**：16:9，推荐 1920×1080 或 3840×2160。  
**语言**：简体中文字幕。  
**语音**：无旁白、无人物语音。  
**字幕**：必须硬编码在画面底部，居中显示；每条字幕必须与对应时间段画面内容一致。  
**音乐**：可使用轻微科技感背景音乐，但音量低，不影响字幕阅读；也可以完全静音。  
**风格**：CVPR / ICCV / NeurIPS 论文方法框架图风格，白色或浅灰背景，扁平矢量图，模块化结构，清晰箭头，少量动画过渡。  
**画面重点**：不要堆砌实验结果，重点讲清方法实现框架：输入、双分支 DiT、Controller、JointConn-v2、GCM-WFM、两种推理模式。

### 1.1 素材与实现口径

- **主参考图**：优先参考项目内框架图 `assets/jointconn_v2_framework.png`，不要重新生成与该图冲突的主架构。
- **骨干网络版本**：画面中明确标注 `FLUX.1-dev style DiT Backbone`，来源为 `black-forest-labs/FLUX.1-dev`。
- **参数规模标注**：可在 Shot 07 右下角小字标注 `about 12B parameters; local transformer count: 11.90B`。
- **实现形式**：当前工程采用 batch-split RGB/depth 逻辑分支。视频可以画成上下双分支，但不要暗示训练了两个独立 FLUX 大模型。
- **主推理任务**：只展示 `joint_generation` 与 `depth_to_image` 两条主线；`depth_estimation` 是兼容旧逻辑，不作为本视频主任务。

---

## 2. 视觉风格规范

### 2.1 颜色规范

- **RGB Branch**：浅蓝色主色，标签使用 `RGB Branch` / `RGB 分支`。
- **Depth Branch**：浅紫色或浅绿色主色，标签使用 `Depth Branch` / `Depth 分支`。
- **Frozen Backbone**：浅灰色模块，标注 `Frozen FLUX.1-dev style DiT Backbone`。
- **Trainable LoRA**：橙色小模块，标注 `LoRA, Trainable`。
- **JointConn-v2**：深蓝色连接模块，放在 RGB 与 Depth 两条分支之间。
- **Geometric Edge Energy**：黄色、红色或金色热力图，用来表示边缘强度。
- **GCM-WFM Loss**：深绿色或蓝绿色模块，表示训练目标。
- **Stop-gradient / detach**：用灰色锁形图标或 `sg(·)` 标签表示。

### 2.2 画面元素

视频中需要反复出现以下核心模块，确保观众能够建立整体印象：

1. `Text Prompt`
2. `CLIP-L Global Text Encoder`
3. `T5 Token Text Encoder`
4. `Frozen VAE Encoder / Decoder`
5. `2×2 Pack / Unpack`
6. `Controller: task, tx, ty, mx, my, λx, λy`
7. `RGB Branch`
8. `Depth Branch`
9. `Frozen FLUX.1-dev style DiT: MM-DiT → P-DiT`
10. `LoRA, Trainable`
11. `JointConn-v2`
12. `Swap-Q Cross-Attention`
13. `Geometric Mask Bias`
14. `Content Gate`
15. `Regional Routing: RGB receives / Depth receives / No Fusion`
16. `Residual Fusion`
17. `GCM-WFM`
18. `α: normalized + clipped + stop-gradient`
19. `Joint Generation: T → RGB + Depth`
20. `Depth-conditioned Generation: T + Depth → RGB`

### 2.3 动画节奏

- 每个核心概念用 8–12 秒介绍。
- 模块图从左到右展开，避免镜头来回跳跃。
- 每次出现新模块时，先高亮模块，再出现字幕。
- 公式只展示关键公式，不要出现过多推导。
- 字幕停留时间必须足够阅读。底部字幕按第 6 节 SRT 固定，不要改写。

---

## 3. 可直接复制给视频生成模型的总 Prompt

```text
Generate a 3-minute academic method-overview video for a computer vision paper titled “JointConn-v2 with GCM-WFM”. The video has no voiceover and no narration. Use hard-coded Simplified Chinese subtitles that exactly match the timeline and visual content. Use the reference architecture image at `assets/jointconn_v2_framework.png` as the main visual reference. The visual style should be a clean CVPR/ICCV/NeurIPS paper architecture animation: white background, flat vector modules, clear arrows, smooth transitions, and readable labels.

The video introduces a unified dual-branch FLUX.1-dev style Diffusion Transformer framework for two tasks: text-conditioned joint RGB-depth generation and depth-conditioned image generation. Show two logical branches: an upper RGB Branch in light blue and a lower Depth Branch in light purple or green. Both branches pass through a shared frozen FLUX.1-dev style DiT backbone with MM-DiT blocks followed by P-DiT blocks. Mark the backbone as frozen, and add a small note: “about 12B parameters; local transformer count: 11.90B”. Add small orange LoRA adapter modules and mark them as trainable. Insert blue JointConn-v2 connector blocks between the two branches at several layers. Do not imply that there are two separately trained FLUX backbones.

At the left side, show the inputs: Text Prompt, RGB latent or RGB noise, and Depth latent or depth condition. The text prompt goes into CLIP-L and T5 text encoders. RGB and depth maps go through a frozen VAE encoder and then a 2×2 Pack module to become token sequences. A Controller module receives the task type, branch timesteps tx and ty, branch masks mx and my, and coupling scales λx and λy. The Controller sends branch-specific time embeddings to the DiT blocks and decides whether the task is joint generation or depth-conditioned generation.

Zoom into JointConn-v2. Inside JointConn-v2, show five submodules: Swap-Q Cross-Attention, Geometric Mask Bias, Content Gate, Regional Routing, and Residual Fusion. Swap-Q Cross-Attention uses bidirectional arrows: RGB tokens query Depth tokens, and Depth tokens query RGB tokens. Geometric Mask Bias adds 2D relative position and depth-edge energy to the attention logits. Visualize the edge energy as a yellow-red edge heatmap aligned with the token grid. Content Gate is a sample-level sigmoid gate. Regional Routing is a token-level router with three choices: RGB receives depth information, Depth receives RGB information, or No Fusion. Residual Fusion injects the selected cross-modal evidence back into each branch with controlled residual arrows.

Then show the training objective GCM-WFM. The two branches output predicted vector fields vx and vy. The loss compares them with reverse-time flow targets τx = εx − zx and τy = εy − zy. Show the diagonal weight α as a heatmap. Explain visually that α combines temporal weight and depth-edge energy, then applies stop-gradient, normalization, and clipping. Emphasize that supervision is stronger around object boundaries and structural regions.

Finally, show the two inference modes. In joint generation mode, RGB and Depth both start from noise and are denoised together, producing a generated RGB image and a generated depth map. In depth-conditioned generation mode, the depth branch remains fixed as a clean condition, while only the RGB branch is denoised to produce the final image.

Use the following exact 18 subtitles and align each subtitle to its 10-second shot. Do not add extra subtitles. Do not use voiceover. Do not include unrelated experimental tables. The final frame should summarize: selective geometry-aware cross-modal communication, stable weighted flow matching, and one unified model for two generation tasks.
```

---

## 4. 分镜设计总表

| 时间 | 画面主题 | 画面内容 | 字幕 |
|---|---|---|---|
| 00:00–00:10 | 标题与任务定位 | 白色背景，标题从中心淡入；下方出现 RGB 图像、深度图和 DiT 图标 | JointConn-v2：几何感知双分支 DiT。 |
| 00:10–00:20 | 两个任务 | 左侧显示文本到 RGB+Depth；右侧显示文本+Depth 到 RGB | 一个模型支持联合生成和深度条件生成。 |
| 00:20–00:30 | 方法动机 | 展示边缘错位、深度拖拽、过耦合三个警示图标 | 边缘易失配，强耦合会扰动深度。 |
| 00:30–00:40 | 核心思想 | RGB 与 Depth 两条分支之间出现可控通信通道 | 我们学习选择性的几何感知通信。 |
| 00:40–00:50 | 输入与 token 化 | Text Prompt、RGB、Depth 分别进入编码器；VAE 与 2×2 Pack 出现 | 图像和深度先编码，再打包为 token。 |
| 00:50–01:00 | Controller | Controller 接收 task、tx、ty、mx、my、λx、λy，并输出时间嵌入 | Controller 分配时间步、掩码和耦合强度。 |
| 01:00–01:10 | 双分支冻结主干 | 上下两条分支进入 Frozen FLUX.1-dev style DiT，内部有 MM-DiT 和 P-DiT | FLUX.1-dev 冻结，只训练适配器。 |
| 01:10–01:20 | JointConn-v2 位置 | 在多个 Transformer block 之间插入蓝色 JointConn-v2 模块 | JointConn-v2 连接双分支。 |
| 01:20–01:30 | Swap-Q Cross-Attention | 放大 JointConn-v2，显示 RGB→Depth 和 Depth→RGB 双向注意力 | Swap-Q 用双向查询交换跨模态信息。 |
| 01:30–01:40 | Geometric Mask Bias | attention logits 上叠加位置 bias 和边缘热力图 | 几何偏置把位置和边缘加入注意力。 |
| 01:40–01:50 | Content Gate | pooled features、timesteps 输入 sigmoid gate，输出 go | Content Gate 控制样本级注入强度。 |
| 01:50–02:00 | Regional Routing | token grid 上出现三类选择：RGB 接收、Depth 接收、No Fusion | Routing 选择 RGB、Depth 或不融合。 |
| 02:00–02:10 | Residual Fusion | 筛选后的信息通过残差箭头回到 RGB 与 Depth 分支 | 残差融合把有效证据注入分支。 |
| 02:10–02:20 | GCM-WFM 损失 | 右侧显示 vx、vy 与 τx、τy 对比，进入 GCM-WFM loss | GCM-WFM 学习双分支向量场。 |
| 02:20–02:30 | 权重 α | α 热力图被标注 stop-gradient、normalized、clipped | α 停止梯度、归一化并裁剪。 |
| 02:30–02:40 | 联合生成推理 | RGB 与 Depth 都从噪声开始，沿时间轴去噪到最终输出 | 联合生成时，RGB 和 Depth 共同去噪。 |
| 02:40–02:50 | 深度条件推理 | Depth 分支固定，RGB 分支从噪声去噪；输出 RGB 图像 | 深度条件时，Depth 固定，只更新 RGB。 |
| 02:50–03:00 | 总结收束 | 三个关键词依次出现：选择性交互、几何边缘、稳定训练 | 最终统一几何通信、加权训练和双任务生成。 |

---

## 5. 每个镜头的详细生成 Prompt

### Shot 01｜00:00–00:10｜标题与任务定位

**画面 Prompt**：

```text
White clean academic background. Center title appears: “JointConn-v2 with GCM-WFM”. Subtitle title below: “Geometry-aware Dual-Branch DiT Framework”. Show three small icons below the title: RGB image thumbnail, grayscale depth map thumbnail, and Transformer block icon. Use blue and purple accent colors. Smooth fade-in animation, no voiceover.
```

**字幕**：

```text
JointConn-v2：几何感知双分支 DiT。
```

---

### Shot 02｜00:10–00:20｜两个任务

**画面 Prompt**：

```text
Split screen into left and right. Left side: Text Prompt arrow points to two outputs, Generated RGB Image and Generated Depth Map, labeled “Joint Generation: T → RGB + Depth”. Right side: Text Prompt plus Depth Condition arrows point to Generated RGB Image, labeled “Depth-conditioned Generation: T + Depth → RGB”. Use clean arrows and simple illustrative thumbnails.
```

**字幕**：

```text
一个模型支持联合生成和深度条件生成。
```

---

### Shot 03｜00:20–00:30｜方法动机

**画面 Prompt**：

```text
Show three problem cards. Card 1: misaligned RGB edge and depth edge, label “Edge mismatch”. Card 2: depth map warped by semantic texture, label “Depth dragging”. Card 3: over-coupled arrows causing artifacts, label “Over-coupling”. Use warning icons, but maintain academic diagram style.
```

**字幕**：

```text
边缘易失配，强耦合会扰动深度。
```

---

### Shot 04｜00:30–00:40｜核心思想

**画面 Prompt**：

```text
Show two horizontal streams: top RGB Branch, bottom Depth Branch. Between them, draw multiple controllable communication bridges with switches. Some switches are open near flat regions, and some are active near edge regions. Highlight the phrase “Selective Geometry-aware Communication”.
```

**字幕**：

```text
我们学习选择性的几何感知通信。
```

---

### Shot 05｜00:40–00:50｜输入与 token 化

**画面 Prompt**：

```text
On the left, show Text Prompt, RGB image or RGB noise, and Depth map or Depth noise. RGB and Depth pass through a Frozen VAE Encoder, then through a “2×2 Pack” block, becoming token grids. Text Prompt goes to two text encoders. Animate image grids turning into square tokens.
```

**字幕**：

```text
图像和深度先编码，再打包为 token。
```

---

### Shot 06｜00:50–01:00｜Controller

**画面 Prompt**：

```text
Show a Controller module on the left side of the two branches. Inputs into Controller: task type, tx, ty, mx, my, λx, λy. Outputs from Controller: branch time embeddings ex and ey, and coupling control arrows to RGB Branch and Depth Branch. Use small clock icons for timesteps and mask icons for branch masks.
```

**字幕**：

```text
Controller 分配时间步、掩码和耦合强度。
```

---

### Shot 07｜01:00–01:10｜双分支冻结主干

**画面 Prompt**：

```text
Show upper RGB Branch and lower Depth Branch passing through one shared large gray frozen backbone labeled “Frozen FLUX.1-dev style DiT Backbone”. Inside it, show MM-DiT blocks followed by P-DiT blocks. Add a small note: “about 12B parameters; local transformer count: 11.90B”. Add small orange LoRA adapter boxes, labeled “LoRA, trainable”. Add lock icons on base backbone weights.
```

**字幕**：

```text
FLUX.1-dev 冻结，只训练适配器。
```

---

### Shot 08｜01:10–01:20｜JointConn-v2 位置

**画面 Prompt**：

```text
Zoom out to the full dual-branch architecture. Insert several blue JointConn-v2 modules between RGB and Depth branches at selected Transformer layers. Animate bidirectional arrows through the connector modules. Label “Inter-branch Connector: JointConn-v2”.
```

**字幕**：

```text
JointConn-v2 连接双分支。
```

---

### Shot 09｜01:20–01:30｜Swap-Q Cross-Attention

**画面 Prompt**：

```text
Zoom into one JointConn-v2 block. Show two token grids: RGB tokens and Depth tokens. Draw bidirectional cross-attention arrows. Label the upper direction “x ← y: RGB queries Depth” and the lower direction “y ← x: Depth queries RGB”. Show Q, K, V icons and an attention matrix.
```

**字幕**：

```text
Swap-Q 用双向查询交换跨模态信息。
```

---

### Shot 10｜01:30–01:40｜Geometric Mask Bias

**画面 Prompt**：

```text
Show the attention matrix receiving two additive bias inputs: “2D Relative Position Bias” and “Edge Energy E”. Visualize E as a yellow-red depth-edge heatmap aligned to the token grid. Highlight object boundary tokens. Show formula label: B = position bias + edge compatibility.
```

**字幕**：

```text
几何偏置把位置和边缘加入注意力。
```

---

### Shot 11｜01:40–01:50｜Content Gate

**画面 Prompt**：

```text
Show pooled RGB features and pooled Depth features, together with tx and ty, entering a small MLP and sigmoid gate. Output a scalar gate go from 0 to 1. The gate modulates the thickness of cross-modal residual arrows. Use “sample-level gate” label.
```

**字幕**：

```text
Content Gate 控制样本级注入强度。
```

---

### Shot 12｜01:50–02:00｜Regional Routing

**画面 Prompt**：

```text
Show a token grid where each token has one of three routing choices. Use three colors and labels: “RGB receives”, “Depth receives”, “No Fusion”. Animate uncertain flat regions choosing No Fusion, and boundary regions choosing cross-modal fusion. Label module as “Token-level Regional Routing”.
```

**字幕**：

```text
Routing 选择 RGB、Depth 或不融合。
```

---

### Shot 13｜02:00–02:10｜Residual Fusion

**画面 Prompt**：

```text
Show outputs from Swap-Q attention passing through gates: time schedule, content gate, regional routing, and branch coupling λ. Then inject the result back into RGB Branch and Depth Branch as controlled residual arrows. Use equation-like labels: R_x and R_y. Keep formulas simple and readable.
```

**字幕**：

```text
残差融合把有效证据注入分支。
```

---

### Shot 14｜02:10–02:20｜GCM-WFM 损失

**画面 Prompt**：

```text
Move to the training objective. Show two predicted vector fields vx and vy from the RGB and Depth branches. Compare them with targets τx = εx − zx and τy = εy − zy. Both enter a green loss block labeled “GCM-WFM: Gated Cross-Modal Weighted Flow Matching”.
```

**字幕**：

```text
GCM-WFM 学习双分支向量场。
```

---

### Shot 15｜02:20–02:30｜权重 α

**画面 Prompt**：

```text
Show diagonal weight α as a token heatmap. Components flow into α: temporal weight and depth-edge energy. Add three badges on α: “stop-gradient”, “normalized”, “clipped”. Highlight stronger weights around object boundaries and structural regions. Use a lock icon to show α is detached.
```

**字幕**：

```text
α 停止梯度、归一化并裁剪。
```

---

### Shot 16｜02:30–02:40｜联合生成推理

**画面 Prompt**：

```text
Show joint generation inference. RGB latent and Depth latent both start as Gaussian noise. A time arrow goes from t=1 to t=0. Both branches are denoised together through the dual-branch model. Final outputs: generated RGB image and generated depth map. Label “T → RGB + Depth”.
```

**字幕**：

```text
联合生成时，RGB 和 Depth 共同去噪。
```

---

### Shot 17｜02:40–02:50｜深度条件推理

**画面 Prompt**：

```text
Show depth-conditioned inference. Depth condition enters the lower branch and stays fixed with a lock icon. RGB branch starts from Gaussian noise and is denoised over time. JointConn-v2 sends geometry guidance from Depth to RGB. Final output is a generated RGB image. Label “T + Depth → RGB”.
```

**字幕**：

```text
深度条件时，Depth 固定，只更新 RGB。
```

---

### Shot 18｜02:50–03:00｜总结收束

**画面 Prompt**：

```text
Show the full architecture again, now simplified and clean. Three key phrases appear one by one: “Selective Cross-modal Communication”, “Geometry-aware Edge Guidance”, “Stable Weighted Flow Matching”. End with title “JointConn-v2 with GCM-WFM” and two output icons: RGB and Depth. Smooth fade out.
```

**字幕**：

```text
最终统一几何通信、加权训练和双任务生成。
```

---

## 6. 字幕文件 SRT

> 下面字幕必须与视频严格对应。若视频生成工具支持导入 SRT，可直接使用此部分。

```srt
1
00:00:00,000 --> 00:00:10,000
JointConn-v2：几何感知双分支 DiT。

2
00:00:10,000 --> 00:00:20,000
一个模型支持联合生成和深度条件生成。

3
00:00:20,000 --> 00:00:30,000
边缘易失配，强耦合会扰动深度。

4
00:00:30,000 --> 00:00:40,000
我们学习选择性的几何感知通信。

5
00:00:40,000 --> 00:00:50,000
图像和深度先编码，再打包为 token。

6
00:00:50,000 --> 00:01:00,000
Controller 分配时间步、掩码和耦合强度。

7
00:01:00,000 --> 00:01:10,000
FLUX.1-dev 冻结，只训练适配器。

8
00:01:10,000 --> 00:01:20,000
JointConn-v2 连接双分支。

9
00:01:20,000 --> 00:01:30,000
Swap-Q 用双向查询交换跨模态信息。

10
00:01:30,000 --> 00:01:40,000
几何偏置把位置和边缘加入注意力。

11
00:01:40,000 --> 00:01:50,000
Content Gate 控制样本级注入强度。

12
00:01:50,000 --> 00:02:00,000
Routing 选择 RGB、Depth 或不融合。

13
00:02:00,000 --> 00:02:10,000
残差融合把有效证据注入分支。

14
00:02:10,000 --> 00:02:20,000
GCM-WFM 学习双分支向量场。

15
00:02:20,000 --> 00:02:30,000
α 停止梯度、归一化并裁剪。

16
00:02:30,000 --> 00:02:40,000
联合生成时，RGB 和 Depth 共同去噪。

17
00:02:40,000 --> 00:02:50,000
深度条件时，Depth 固定，只更新 RGB。

18
00:02:50,000 --> 00:03:00,000
最终统一几何通信、加权训练和双任务生成。
```

---

## 7. 屏幕文字与公式建议

为了避免视频画面拥挤，公式只展示以下 4 个核心表达式：

### 7.1 反向流路径

```latex
z_t^e=(1-t)z_{data}^e+t\epsilon^e
```

出现位置：Shot 14 前后或 Shot 05 的 token 化之后。  
视觉解释：从噪声到数据的时间轴，`t=1` 是 noise，`t=0` 是 data。

### 7.2 速度目标

```latex
\tau_e=\epsilon^e-z_{data}^e
```

出现位置：Shot 14。  
视觉解释：预测向量场与目标向量场对齐。

### 7.3 几何注意力偏置

```latex
B_{ij}=\Pi(p_i-p_j)+\beta E_iE_jK_\sigma(p_i,p_j)
```

出现位置：Shot 10。  
视觉解释：位置相近且边缘强的 token pair 会被更强关注。

### 7.4 加权损失

```latex
\mathcal{L}=\sum_e m_e\,\alpha_e\|\hat v_e-\tau_e\|^2
```

出现位置：Shot 14–15。  
视觉解释：不同 token 有不同监督权重，结构边缘区域权重更高。

---

## 8. 镜头转场建议

- **Shot 01 → Shot 02**：标题缩小到左上角，任务图从中间展开。
- **Shot 02 → Shot 03**：任务图淡出，问题卡片从左右滑入。
- **Shot 03 → Shot 04**：问题卡片变成两条分支之间的通信桥。
- **Shot 04 → Shot 08**：逐步搭建完整架构图。
- **Shot 09 → Shot 13**：连续 zoom-in 到 JointConn-v2 内部，每个子模块依次高亮。
- **Shot 14 → Shot 15**：从模型输出平移到训练损失，α 热力图从边缘区域亮起。
- **Shot 16 → Shot 17**：左右分屏展示两种推理模式。
- **Shot 17 → Shot 18**：回到完整框架图，总结关键词依次出现。

---

## 9. 字幕排版规则

- 字幕位置：画面底部居中，距离底边约 6% 高度。
- 字体：思源黑体、苹方、Noto Sans CJK 或等价无衬线字体。
- 字号：1080p 推荐 38–44 px；4K 推荐 76–88 px。
- 字幕颜色：白色文字 + 35% 透明黑色圆角背景条。
- 每条字幕最多两行。
- 不要自动改写字幕，不要额外添加英文字幕。
- 模块标签可以使用英文，但底部字幕必须是上述中文句子。

---

## 10. 负面 Prompt / 禁止事项

```text
Do not add voiceover. Do not add talking humans. Do not add unrelated benchmark tables. Do not show excessive equations. Do not use messy 3D sci-fi graphics. Do not use dark cyberpunk background. Do not make the subtitles different from the given SRT. Do not show RGB-to-depth depth_estimation as a main task. Do not show the depth branch being updated in depth-conditioned mode. Do not draw two independently trained FLUX backbones. Do not mark the frozen backbone as trainable. Do not omit LoRA. Do not omit JointConn-v2. Do not omit the no-fusion routing state. Do not show gate/routing as mandatory inputs to α unless explicitly labeled as optional detached reliability. Do not use random decorative neural network graphics that do not correspond to the described modules.
```

---

## 11. 最终检查清单

生成视频前后请逐项检查：

- [ ] 总时长约 3 分钟。
- [ ] 没有语音或旁白。
- [ ] 字幕完整出现，且与 SRT 一致。
- [ ] 使用或参考 `assets/jointconn_v2_framework.png` 作为主框架图。
- [ ] 画面主线是方法实现框架，而不是实验结果。
- [ ] 出现两个任务：联合生成与深度条件生成。
- [ ] 出现双分支 RGB / Depth DiT 架构。
- [ ] 主干明确标注为 `Frozen FLUX.1-dev style DiT Backbone`。
- [ ] 画面小字说明 `about 12B parameters; local transformer count: 11.90B`。
- [ ] 没有画成两个独立训练的 FLUX 主干。
- [ ] LoRA 明确标注为 Trainable。
- [ ] JointConn-v2 出现在两条分支之间。
- [ ] JointConn-v2 内部包含 Swap-Q、Geometric Mask Bias、Content Gate、Regional Routing、Residual Fusion。
- [ ] Regional Routing 明确包含 No Fusion 选项。
- [ ] GCM-WFM 中出现向量场回归和权重 α。
- [ ] α 明确标注 stop-gradient、normalized、clipped。
- [ ] α 主要由时间权重和 depth edge energy 构成，gate/routing 不画成必选输入。
- [ ] depth-conditioned 模式中 Depth 分支保持固定，只更新 RGB 分支。
- [ ] 没有把旧的 `depth_estimation` 兼容模式作为主任务展示。
- [ ] 最后一幕总结“几何通信、加权训练、双任务生成”。
