import type { Caption } from "@remotion/captions";

export const captions: Caption[] = [
  { text: "JointConn-v2：几何感知双分支 DiT。", startMs: 0, endMs: 10000, timestampMs: null, confidence: null },
  { text: "一个模型支持联合生成和深度条件生成。", startMs: 10000, endMs: 20000, timestampMs: null, confidence: null },
  { text: "边缘易失配，强耦合会扰动深度。", startMs: 20000, endMs: 30000, timestampMs: null, confidence: null },
  { text: "我们学习选择性的几何感知通信。", startMs: 30000, endMs: 40000, timestampMs: null, confidence: null },
  { text: "图像和深度先编码，再打包为 token。", startMs: 40000, endMs: 50000, timestampMs: null, confidence: null },
  { text: "Controller 分配时间步、掩码和耦合强度。", startMs: 50000, endMs: 60000, timestampMs: null, confidence: null },
  { text: "FLUX.1-dev 冻结，只训练适配器。", startMs: 60000, endMs: 70000, timestampMs: null, confidence: null },
  { text: "JointConn-v2 连接双分支。", startMs: 70000, endMs: 80000, timestampMs: null, confidence: null },
  { text: "Swap-Q 用双向查询交换跨模态信息。", startMs: 80000, endMs: 90000, timestampMs: null, confidence: null },
  { text: "几何偏置把位置和边缘加入注意力。", startMs: 90000, endMs: 100000, timestampMs: null, confidence: null },
  { text: "Content Gate 控制样本级注入强度。", startMs: 100000, endMs: 110000, timestampMs: null, confidence: null },
  { text: "Routing 选择 RGB、Depth 或不融合。", startMs: 110000, endMs: 120000, timestampMs: null, confidence: null },
  { text: "残差融合把有效证据注入分支。", startMs: 120000, endMs: 130000, timestampMs: null, confidence: null },
  { text: "GCM-WFM 学习双分支向量场。", startMs: 130000, endMs: 140000, timestampMs: null, confidence: null },
  { text: "α 停止梯度、归一化并裁剪。", startMs: 140000, endMs: 150000, timestampMs: null, confidence: null },
  { text: "联合生成时，RGB 和 Depth 共同去噪。", startMs: 150000, endMs: 160000, timestampMs: null, confidence: null },
  { text: "深度条件时，Depth 固定，只更新 RGB。", startMs: 160000, endMs: 170000, timestampMs: null, confidence: null },
  { text: "最终统一几何通信、加权训练和双任务生成。", startMs: 170000, endMs: 180000, timestampMs: null, confidence: null },
];
