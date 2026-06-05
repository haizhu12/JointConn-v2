import type { CSSProperties, ReactNode } from "react";
import {
  AbsoluteFill,
  Easing,
  Img,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { captions } from "./captions";

const SCENE_SECONDS = 10;
const SCENE_COUNT = 18;
const BLUE = "#2563eb";
const LIGHT_BLUE = "#dbeafe";
const PURPLE = "#7c3aed";
const LIGHT_PURPLE = "#ede9fe";
const GREEN = "#047857";
const ORANGE = "#f97316";
const YELLOW = "#facc15";
const INK = "#172033";
const MUTED = "#687385";

type BoxProps = {
  children: ReactNode;
  color?: string;
  fill?: string;
  style?: CSSProperties;
};

const ease = Easing.bezier(0.16, 1, 0.3, 1);

const scenes = [
  "Title",
  "Two Tasks",
  "Motivation",
  "Core Idea",
  "Tokenization",
  "Controller",
  "Frozen Backbone",
  "Connector Sites",
  "Swap-Q Attention",
  "Geometry Bias",
  "Content Gate",
  "Regional Routing",
  "Residual Fusion",
  "GCM-WFM",
  "Alpha Weight",
  "Joint Inference",
  "Depth Condition",
  "Summary",
];

const baseFont: CSSProperties = {
  fontFamily:
    '"Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", Arial, sans-serif',
};

const fadeStyle = (localFrame: number): CSSProperties => ({
  opacity: interpolate(localFrame, [0, 22, 278, 300], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: ease,
  }),
  transform: `translateY(${interpolate(localFrame, [0, 26], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: ease,
  })}px)`,
});

const Box = ({ children, color = BLUE, fill = "#fff", style }: BoxProps) => (
  <div
    style={{
      ...baseFont,
      border: `3px solid ${color}`,
      background: fill,
      color: INK,
      borderRadius: 8,
      padding: "18px 22px",
      boxShadow: "0 10px 28px rgba(23, 32, 51, 0.08)",
      fontSize: 30,
      fontWeight: 700,
      textAlign: "center",
      ...style,
    }}
  >
    {children}
  </div>
);

const Label = ({ children, style }: { children: ReactNode; style?: CSSProperties }) => (
  <div
    style={{
      ...baseFont,
      color: MUTED,
      fontSize: 25,
      fontWeight: 700,
      letterSpacing: 0,
      ...style,
    }}
  >
    {children}
  </div>
);

const Arrow = ({
  x1,
  y1,
  x2,
  y2,
  color = "#334155",
  width = 5,
  delay = 0,
  localFrame,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color?: string;
  width?: number;
  delay?: number;
  localFrame: number;
}) => {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  const progress = interpolate(localFrame, [delay, delay + 34], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: ease,
  });

  return (
    <div
      style={{
        position: "absolute",
        left: x1,
        top: y1,
        width: length * progress,
        height: width,
        background: color,
        borderRadius: width,
        transformOrigin: "0 50%",
        transform: `rotate(${angle}deg)`,
      }}
    >
      <div
        style={{
          position: "absolute",
          right: -2,
          top: -7,
          width: 0,
          height: 0,
          borderTop: "9px solid transparent",
          borderBottom: "9px solid transparent",
          borderLeft: `18px solid ${color}`,
          opacity: progress > 0.92 ? 1 : 0,
        }}
      />
    </div>
  );
};

const TokenGrid = ({
  colors,
  rows = 5,
  cols = 8,
  size = 26,
  gap = 7,
  style,
}: {
  colors: string[];
  rows?: number;
  cols?: number;
  size?: number;
  gap?: number;
  style?: CSSProperties;
}) => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: `repeat(${cols}, ${size}px)`,
      gap,
      ...style,
    }}
  >
    {Array.from({ length: rows * cols }).map((_, i) => (
      <div
        key={i}
        style={{
          width: size,
          height: size,
          borderRadius: 5,
          background: colors[i % colors.length],
          border: "1px solid rgba(23,32,51,0.12)",
        }}
      />
    ))}
  </div>
);

const Heatmap = ({ style }: { style?: CSSProperties }) => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "repeat(9, 28px)",
      gap: 6,
      ...style,
    }}
  >
    {Array.from({ length: 54 }).map((_, i) => {
      const edge =
        i % 9 === 2 || i % 9 === 6 || Math.floor(i / 9) === 2 || Math.floor(i / 9) === 4;
      return (
        <div
          key={i}
          style={{
            width: 28,
            height: 28,
            borderRadius: 5,
            background: edge ? (i % 2 ? "#ef4444" : "#facc15") : "#fef9c3",
            boxShadow: edge ? "0 0 18px rgba(239, 68, 68, 0.35)" : "none",
          }}
        />
      );
    })}
  </div>
);

const Header = ({ sceneIndex }: { sceneIndex: number }) => (
  <div
    style={{
      ...baseFont,
      position: "absolute",
      left: 64,
      right: 64,
      top: 36,
      height: 60,
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      color: INK,
      zIndex: 20,
    }}
  >
    <div style={{ fontSize: 24, fontWeight: 800, color: "#334155" }}>
      JointConn-v2 with GCM-WFM
    </div>
    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
      <div
        style={{
          width: 230,
          height: 8,
          borderRadius: 10,
          background: "#e2e8f0",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${((sceneIndex + 1) / SCENE_COUNT) * 100}%`,
            height: "100%",
            background: `linear-gradient(90deg, ${BLUE}, ${GREEN})`,
          }}
        />
      </div>
      <div style={{ color: MUTED, fontSize: 22, fontWeight: 700 }}>
        {String(sceneIndex + 1).padStart(2, "0")} / 18 · {scenes[sceneIndex]}
      </div>
    </div>
  </div>
);

const CaptionBar = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const nowMs = (frame / fps) * 1000;
  const caption = captions.find((item) => nowMs >= item.startMs && nowMs < item.endMs);

  return (
    <div
      style={{
        ...baseFont,
        position: "absolute",
        left: 260,
        right: 260,
        bottom: 52,
        minHeight: 72,
        borderRadius: 8,
        background: "rgba(15, 23, 42, 0.72)",
        color: "#fff",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        fontSize: 42,
        fontWeight: 800,
        lineHeight: 1.25,
        padding: "10px 34px",
        zIndex: 30,
      }}
    >
      {caption?.text}
    </div>
  );
};

const Background = () => (
  <AbsoluteFill
    style={{
      background:
        "linear-gradient(180deg, #ffffff 0%, #f8fafc 58%, #eef6ff 100%)",
      overflow: "hidden",
    }}
  >
    <div
      style={{
        position: "absolute",
        inset: 0,
        backgroundImage:
          "linear-gradient(rgba(148,163,184,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.12) 1px, transparent 1px)",
        backgroundSize: "48px 48px",
      }}
    />
  </AbsoluteFill>
);

const SceneShell = ({ children, localFrame }: { children: ReactNode; localFrame: number }) => (
  <AbsoluteFill style={{ ...fadeStyle(localFrame), zIndex: 5 }}>{children}</AbsoluteFill>
);

const Branch = ({
  label,
  color,
  y,
  localFrame,
}: {
  label: string;
  color: string;
  y: number;
  localFrame: number;
}) => (
  <>
    <Box
      color={color}
      fill={color === BLUE ? LIGHT_BLUE : LIGHT_PURPLE}
      style={{ position: "absolute", left: 250, top: y, width: 230 }}
    >
      {label}
    </Box>
    <Box
      color="#94a3b8"
      fill="#f1f5f9"
      style={{ position: "absolute", left: 620, top: y - 16, width: 640, height: 96 }}
    >
      Frozen FLUX.1-dev style DiT
    </Box>
    <Arrow x1={500} y1={y + 48} x2={620} y2={y + 48} color={color} localFrame={localFrame} />
    <Arrow x1={1260} y1={y + 48} x2={1400} y2={y + 48} color={color} localFrame={localFrame} />
  </>
);

const Connector = ({ top, left, localFrame }: { top: number; left: number; localFrame: number }) => {
  const pulse = interpolate(localFrame % 60, [0, 30, 60], [0.65, 1, 0.65], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        ...baseFont,
        position: "absolute",
        left,
        top,
        width: 190,
        height: 98,
        borderRadius: 8,
        background: BLUE,
        color: "#fff",
        fontSize: 26,
        fontWeight: 800,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        boxShadow: `0 0 ${18 * pulse}px rgba(37,99,235,0.45)`,
      }}
    >
      JointConn-v2
    </div>
  );
};

const TitleScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <div style={{ position: "absolute", left: 110, top: 126, width: 820 }}>
      <div
        style={{
          ...baseFont,
          color: INK,
          fontSize: 82,
          fontWeight: 900,
          lineHeight: 1.05,
        }}
      >
        JointConn-v2
      </div>
      <div
        style={{
          ...baseFont,
          color: GREEN,
          fontSize: 45,
          fontWeight: 850,
          marginTop: 14,
        }}
      >
        Geometry-aware Dual-Branch DiT Framework
      </div>
      <div
        style={{
          ...baseFont,
          color: MUTED,
          fontSize: 30,
          lineHeight: 1.4,
          marginTop: 34,
          width: 740,
        }}
      >
        A unified framework for RGB-depth joint generation and
        depth-conditioned image generation.
      </div>
    </div>
    <Img
      src={staticFile("jointconn_v2_framework.png")}
      style={{
        position: "absolute",
        right: 92,
        top: 120,
        width: 810,
        height: 560,
        objectFit: "contain",
        borderRadius: 8,
        boxShadow: "0 24px 60px rgba(15, 23, 42, 0.16)",
        border: "1px solid #dbe3ef",
        background: "#fff",
      }}
    />
    <div style={{ position: "absolute", left: 110, top: 560, display: "flex", gap: 22 }}>
      <Box color={BLUE} fill={LIGHT_BLUE} style={{ width: 170 }}>
        RGB
      </Box>
      <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ width: 170 }}>
        Depth
      </Box>
      <Box color={GREEN} fill="#dcfce7" style={{ width: 170 }}>
        DiT
      </Box>
    </div>
  </SceneShell>
);

const TasksScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <div style={{ position: "absolute", left: 120, top: 150, width: 780, height: 560 }}>
      <Label style={{ fontSize: 34, color: BLUE }}>Joint Generation</Label>
      <Box color="#475569" style={{ position: "absolute", left: 20, top: 120, width: 230 }}>
        Text Prompt
      </Box>
      <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", left: 500, top: 70, width: 210 }}>
        RGB Image
      </Box>
      <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", left: 500, top: 230, width: 210 }}>
        Depth Map
      </Box>
      <Arrow x1={260} y1={168} x2={500} y2={126} color={BLUE} localFrame={localFrame} />
      <Arrow x1={260} y1={188} x2={500} y2={286} color={PURPLE} localFrame={localFrame} />
      <Label style={{ position: "absolute", left: 280, top: 390, fontSize: 38 }}>
        T → RGB + Depth
      </Label>
    </div>
    <div style={{ position: "absolute", right: 120, top: 150, width: 780, height: 560 }}>
      <Label style={{ fontSize: 34, color: GREEN }}>Depth-conditioned Generation</Label>
      <Box color="#475569" style={{ position: "absolute", left: 20, top: 90, width: 230 }}>
        Text Prompt
      </Box>
      <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", left: 20, top: 245, width: 230 }}>
        Depth Condition
      </Box>
      <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", left: 510, top: 168, width: 220 }}>
        RGB Image
      </Box>
      <Arrow x1={260} y1={137} x2={510} y2={210} color={BLUE} localFrame={localFrame} />
      <Arrow x1={260} y1={292} x2={510} y2={250} color={PURPLE} localFrame={localFrame} />
      <Label style={{ position: "absolute", left: 265, top: 390, fontSize: 38 }}>
        T + Depth → RGB
      </Label>
    </div>
  </SceneShell>
);

const MotivationScene = ({ localFrame }: { localFrame: number }) => {
  const cards = [
    ["Edge mismatch", "RGB edge and depth edge drift apart", "#fee2e2"],
    ["Depth dragging", "Texture noise contaminates geometry", "#fef3c7"],
    ["Over-coupling", "Every token talks even when unreliable", "#e0f2fe"],
  ];
  return (
    <SceneShell localFrame={localFrame}>
      <div
        style={{
          ...baseFont,
          position: "absolute",
          left: 120,
          top: 130,
          color: INK,
          fontSize: 56,
          fontWeight: 900,
        }}
      >
        Why selective communication?
      </div>
      <div style={{ position: "absolute", left: 120, right: 120, top: 255, display: "flex", gap: 36 }}>
        {cards.map(([title, desc, fill], index) => (
          <div
            key={title}
            style={{
              width: 520,
              height: 360,
              borderRadius: 8,
              background: fill,
              border: "2px solid rgba(15,23,42,0.1)",
              padding: 34,
              boxShadow: "0 18px 45px rgba(15,23,42,0.10)",
              transform: `translateY(${interpolate(localFrame, [index * 10, index * 10 + 24], [24, 0], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
                easing: ease,
              })}px)`,
              opacity: interpolate(localFrame, [index * 10, index * 10 + 24], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            <div style={{ fontSize: 74, marginBottom: 22 }}>!</div>
            <div style={{ ...baseFont, color: INK, fontSize: 39, fontWeight: 900 }}>{title}</div>
            <div style={{ ...baseFont, color: MUTED, fontSize: 28, lineHeight: 1.32, marginTop: 24 }}>
              {desc}
            </div>
          </div>
        ))}
      </div>
    </SceneShell>
  );
};

const CoreIdeaScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Branch label="RGB Branch" color={BLUE} y={250} localFrame={localFrame} />
    <Branch label="Depth Branch" color={PURPLE} y={500} localFrame={localFrame} />
    {[640, 865, 1090].map((left, i) => (
      <Connector key={left} left={left} top={385} localFrame={localFrame + i * 12} />
    ))}
    {[690, 915, 1140].map((x, i) => (
      <div key={x}>
        <Arrow x1={x} y1={356} x2={x} y2={500} color={GREEN} delay={i * 8} localFrame={localFrame} />
        <Arrow x1={x + 70} y1={500} x2={x + 70} y2={356} color={BLUE} delay={i * 8 + 12} localFrame={localFrame} />
      </div>
    ))}
    <Label style={{ position: "absolute", left: 590, top: 165, fontSize: 44, color: INK }}>
      Selective Geometry-aware Communication
    </Label>
  </SceneShell>
);

const TokenizationScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color="#475569" style={{ position: "absolute", left: 120, top: 205, width: 220 }}>
      Text Prompt
    </Box>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", left: 120, top: 385, width: 220 }}>
      RGB / Noise
    </Box>
    <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", left: 120, top: 565, width: 220 }}>
      Depth / Cond.
    </Box>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 520, top: 170, width: 260 }}>
      CLIP-L
    </Box>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 520, top: 290, width: 260 }}>
      T5 Tokens
    </Box>
    <Box color="#94a3b8" fill="#f1f5f9" style={{ position: "absolute", left: 520, top: 475, width: 260 }}>
      Frozen VAE
    </Box>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 920, top: 475, width: 260 }}>
      2×2 Pack
    </Box>
    <TokenGrid
      colors={["#dbeafe", "#bfdbfe", "#ede9fe", "#dcfce7"]}
      style={{ position: "absolute", left: 1340, top: 424 }}
      size={34}
      gap={9}
    />
    <Arrow x1={350} y1={244} x2={520} y2={220} color="#475569" localFrame={localFrame} />
    <Arrow x1={350} y1={244} x2={520} y2={340} color="#475569" localFrame={localFrame} />
    <Arrow x1={350} y1={430} x2={520} y2={520} color={BLUE} localFrame={localFrame} />
    <Arrow x1={350} y1={610} x2={520} y2={540} color={PURPLE} localFrame={localFrame} />
    <Arrow x1={780} y1={525} x2={920} y2={525} color={GREEN} localFrame={localFrame} />
    <Arrow x1={1180} y1={525} x2={1340} y2={525} color={GREEN} localFrame={localFrame} />
  </SceneShell>
);

const ControllerScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 700, top: 330, width: 410, height: 150 }}>
      Controller
      <div style={{ fontSize: 22, color: MUTED, marginTop: 12 }}>task, tx, ty, mx, my, λx, λy</div>
    </Box>
    {["task", "tx", "ty", "mx", "my", "λx", "λy"].map((item, index) => (
      <Box
        key={item}
        color="#64748b"
        fill="#f8fafc"
        style={{
          position: "absolute",
          left: 125 + index * 138,
          top: index % 2 ? 180 : 640,
          width: 92,
          fontSize: 26,
          padding: "16px 10px",
        }}
      >
        {item}
      </Box>
    ))}
    <Branch label="RGB Branch" color={BLUE} y={185} localFrame={localFrame} />
    <Branch label="Depth Branch" color={PURPLE} y={590} localFrame={localFrame} />
    <Arrow x1={700} y1={410} x2={510} y2={233} color={BLUE} localFrame={localFrame} />
    <Arrow x1={700} y1={430} x2={510} y2={638} color={PURPLE} localFrame={localFrame} />
    <Arrow x1={1110} y1={390} x2={1285} y2={233} color={BLUE} localFrame={localFrame} />
    <Arrow x1={1110} y1={430} x2={1285} y2={638} color={PURPLE} localFrame={localFrame} />
  </SceneShell>
);

const BackboneScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <div
      style={{
        position: "absolute",
        left: 210,
        top: 185,
        width: 1500,
        height: 455,
        borderRadius: 8,
        background: "#f1f5f9",
        border: "3px solid #94a3b8",
        boxShadow: "0 24px 60px rgba(15,23,42,0.12)",
      }}
    />
    <Label style={{ position: "absolute", left: 255, top: 220, color: INK, fontSize: 42 }}>
      Frozen FLUX.1-dev style DiT Backbone
    </Label>
    <Label style={{ position: "absolute", right: 255, top: 225, fontSize: 25 }}>
      about 12B parameters · local transformer count: 11.90B
    </Label>
    <div style={{ position: "absolute", left: 280, top: 330, display: "flex", gap: 22 }}>
      {Array.from({ length: 7 }).map((_, i) => (
        <Box
          key={i}
          color={i < 3 ? BLUE : GREEN}
          fill={i < 3 ? LIGHT_BLUE : "#dcfce7"}
          style={{ width: 150, height: 100, fontSize: 25, padding: 16 }}
        >
          {i < 3 ? "MM-DiT" : "P-DiT"}
        </Box>
      ))}
    </div>
    {[390, 590, 790, 990, 1190, 1390].map((left, i) => (
      <Box
        key={left}
        color={ORANGE}
        fill="#ffedd5"
        style={{
          position: "absolute",
          left,
          top: i % 2 ? 520 : 470,
          width: 110,
          fontSize: 22,
          padding: 12,
        }}
      >
        LoRA
      </Box>
    ))}
    <div
      style={{
        ...baseFont,
        position: "absolute",
        left: 265,
        bottom: 255,
        color: "#64748b",
        fontSize: 27,
        fontWeight: 700,
      }}
    >
      Base VAE, CLIP-L, T5-XXL and transformer weights are locked.
    </div>
  </SceneShell>
);

const ConnectorSitesScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Branch label="RGB Branch" color={BLUE} y={250} localFrame={localFrame} />
    <Branch label="Depth Branch" color={PURPLE} y={535} localFrame={localFrame} />
    {[650, 850, 1050, 1250].map((left, i) => (
      <Connector key={left} left={left} top={400} localFrame={localFrame + i * 6} />
    ))}
    <Label style={{ position: "absolute", left: 610, top: 158, fontSize: 46, color: INK }}>
      JointConn-v2 at selected layers
    </Label>
  </SceneShell>
);

const SwapQScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", left: 230, top: 250, width: 360 }}>
      RGB Tokens
      <TokenGrid colors={["#dbeafe", "#93c5fd"]} style={{ margin: "24px auto 0" }} />
    </Box>
    <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", right: 230, top: 250, width: 360 }}>
      Depth Tokens
      <TokenGrid colors={["#ede9fe", "#c4b5fd"]} style={{ margin: "24px auto 0" }} />
    </Box>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 740, top: 300, width: 430, height: 190 }}>
      Swap-Q Cross-Attention
      <div style={{ fontSize: 24, color: MUTED, marginTop: 20 }}>QxKyᵀ and QyKxᵀ</div>
    </Box>
    <Arrow x1={590} y1={350} x2={740} y2={350} color={BLUE} localFrame={localFrame} />
    <Arrow x1={1170} y1={430} x2={1340} y2={430} color={PURPLE} localFrame={localFrame} />
    <Arrow x1={1340} y1={350} x2={1170} y2={350} color={PURPLE} delay={18} localFrame={localFrame} />
    <Arrow x1={740} y1={430} x2={590} y2={430} color={BLUE} delay={18} localFrame={localFrame} />
  </SceneShell>
);

const GeometryBiasScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color="#475569" fill="#f8fafc" style={{ position: "absolute", left: 150, top: 270, width: 430, height: 280 }}>
      Attention Logits
      <TokenGrid colors={["#e2e8f0", "#cbd5e1", "#f8fafc"]} rows={6} cols={7} style={{ margin: "28px auto 0" }} />
    </Box>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 760, top: 195, width: 350 }}>
      2D Relative Position
    </Box>
    <Box color={YELLOW} fill="#fef9c3" style={{ position: "absolute", left: 760, top: 405, width: 350 }}>
      Edge Energy E
      <Heatmap style={{ margin: "20px auto 0", transform: "scale(0.72)", transformOrigin: "top center" }} />
    </Box>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", right: 150, top: 315, width: 430 }}>
      Bᵢⱼ = Π(pᵢ-pⱼ) + βEᵢEⱼKσ
    </Box>
    <Arrow x1={580} y1={408} x2={760} y2={250} color={GREEN} localFrame={localFrame} />
    <Arrow x1={580} y1={430} x2={760} y2={475} color={YELLOW} delay={10} localFrame={localFrame} />
    <Arrow x1={1110} y1={380} x2={1340} y2={380} color={BLUE} delay={20} localFrame={localFrame} />
  </SceneShell>
);

const ContentGateScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", left: 170, top: 230, width: 280 }}>
      GAP(RGB)
    </Box>
    <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", left: 170, top: 430, width: 280 }}>
      GAP(Depth)
    </Box>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 170, top: 620, width: 280 }}>
      tx, ty, layer
    </Box>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 710, top: 350, width: 380, height: 170 }}>
      MLP + Sigmoid
      <div style={{ fontSize: 56, color: GREEN, marginTop: 12 }}>gₒ</div>
    </Box>
    <Box color={ORANGE} fill="#ffedd5" style={{ position: "absolute", right: 200, top: 345, width: 350 }}>
      Sample-level Gate
      <div style={{ fontSize: 24, color: MUTED, marginTop: 16 }}>controls residual strength</div>
    </Box>
    {[275, 475, 665].map((y, i) => (
      <Arrow key={y} x1={450} y1={y} x2={710} y2={425} color={i === 0 ? BLUE : i === 1 ? PURPLE : "#64748b"} localFrame={localFrame} />
    ))}
    <Arrow x1={1090} y1={435} x2={1370} y2={435} color={ORANGE} delay={18} width={9} localFrame={localFrame} />
  </SceneShell>
);

const RoutingScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Label style={{ position: "absolute", left: 126, top: 140, color: INK, fontSize: 46 }}>
      Token-level Regional Routing
    </Label>
    <TokenGrid
      colors={["#dbeafe", "#ede9fe", "#e5e7eb", "#dbeafe", "#e5e7eb", "#ede9fe"]}
      rows={8}
      cols={12}
      size={42}
      gap={10}
      style={{ position: "absolute", left: 300, top: 255 }}
    />
    <div style={{ position: "absolute", right: 220, top: 255, display: "grid", gap: 28 }}>
      <Box color={BLUE} fill={LIGHT_BLUE} style={{ width: 420 }}>
        RGB receives
      </Box>
      <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ width: 420 }}>
        Depth receives
      </Box>
      <Box color="#94a3b8" fill="#f1f5f9" style={{ width: 420 }}>
        No Fusion
      </Box>
    </div>
  </SceneShell>
);

const ResidualFusionScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 135, top: 360, width: 310 }}>
      Cross-modal Evidence
    </Box>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 590, top: 215, width: 270 }}>
      Time Schedule
    </Box>
    <Box color={ORANGE} fill="#ffedd5" style={{ position: "absolute", left: 590, top: 360, width: 270 }}>
      Content Gate
    </Box>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", left: 590, top: 505, width: 270 }}>
      Routing + λ
    </Box>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", right: 190, top: 255, width: 330 }}>
      RGB Residual Rₓ
    </Box>
    <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", right: 190, top: 500, width: 330 }}>
      Depth Residual Rᵧ
    </Box>
    <Arrow x1={445} y1={410} x2={590} y2={260} color="#64748b" localFrame={localFrame} />
    <Arrow x1={445} y1={410} x2={590} y2={410} color={ORANGE} delay={8} localFrame={localFrame} />
    <Arrow x1={445} y1={410} x2={590} y2={550} color={BLUE} delay={16} localFrame={localFrame} />
    <Arrow x1={860} y1={410} x2={1400} y2={310} color={BLUE} delay={24} width={8} localFrame={localFrame} />
    <Arrow x1={860} y1={430} x2={1400} y2={555} color={PURPLE} delay={30} width={8} localFrame={localFrame} />
  </SceneShell>
);

const LossScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", left: 160, top: 260, width: 300 }}>
      v̂ₓ
      <div style={{ fontSize: 24, color: MUTED, marginTop: 12 }}>RGB vector field</div>
    </Box>
    <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", left: 160, top: 480, width: 300 }}>
      v̂ᵧ
      <div style={{ fontSize: 24, color: MUTED, marginTop: 12 }}>Depth vector field</div>
    </Box>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 665, top: 260, width: 360 }}>
      τₓ = εₓ - zₓ
    </Box>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 665, top: 480, width: 360 }}>
      τᵧ = εᵧ - zᵧ
    </Box>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", right: 160, top: 350, width: 430, height: 170 }}>
      GCM-WFM Loss
      <div style={{ fontSize: 24, color: MUTED, marginTop: 14 }}>branch-masked packed-token loss</div>
    </Box>
    <Arrow x1={460} y1={310} x2={665} y2={310} color={BLUE} localFrame={localFrame} />
    <Arrow x1={460} y1={530} x2={665} y2={530} color={PURPLE} localFrame={localFrame} />
    <Arrow x1={1025} y1={310} x2={1330} y2={405} color={GREEN} delay={16} localFrame={localFrame} />
    <Arrow x1={1025} y1={530} x2={1330} y2={470} color={GREEN} delay={16} localFrame={localFrame} />
  </SceneShell>
);

const AlphaScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 160, top: 260, width: 330 }}>
      Temporal Weight
      <div style={{ fontSize: 42, color: "#64748b", marginTop: 20 }}>w(t)</div>
    </Box>
    <Box color={YELLOW} fill="#fef9c3" style={{ position: "absolute", left: 160, top: 500, width: 330 }}>
      Depth Edge Energy
      <div style={{ fontSize: 42, color: "#ca8a04", marginTop: 20 }}>E</div>
    </Box>
    <Heatmap style={{ position: "absolute", left: 720, top: 250, transform: "scale(1.25)", transformOrigin: "top left" }} />
    <Label style={{ position: "absolute", left: 755, top: 585, color: INK, fontSize: 54 }}>α</Label>
    <div style={{ position: "absolute", right: 210, top: 250, display: "grid", gap: 26 }}>
      {["stop-gradient", "normalized", "clipped"].map((item) => (
        <Box key={item} color={GREEN} fill="#dcfce7" style={{ width: 360 }}>
          {item}
        </Box>
      ))}
    </div>
    <Arrow x1={490} y1={320} x2={700} y2={360} color="#64748b" localFrame={localFrame} />
    <Arrow x1={490} y1={560} x2={700} y2={470} color={YELLOW} delay={10} localFrame={localFrame} />
  </SceneShell>
);

const JointInferenceScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Label style={{ position: "absolute", left: 130, top: 150, color: INK, fontSize: 48 }}>
      Joint Generation: T → RGB + Depth
    </Label>
    <Box color="#64748b" fill="#f8fafc" style={{ position: "absolute", left: 155, top: 315, width: 230 }}>
      Text Prompt
    </Box>
    <Box color={BLUE} fill="#eff6ff" style={{ position: "absolute", left: 520, top: 250, width: 210 }}>
      RGB Noise
    </Box>
    <Box color={PURPLE} fill="#f5f3ff" style={{ position: "absolute", left: 520, top: 470, width: 210 }}>
      Depth Noise
    </Box>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 890, top: 345, width: 310 }}>
      Dual-Branch Denoising
      <div style={{ fontSize: 24, color: MUTED, marginTop: 12 }}>t = 1 → 0</div>
    </Box>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", right: 165, top: 250, width: 260 }}>
      Generated RGB
    </Box>
    <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", right: 165, top: 470, width: 260 }}>
      Generated Depth
    </Box>
    <Arrow x1={385} y1={365} x2={520} y2={310} color="#64748b" localFrame={localFrame} />
    <Arrow x1={730} y1={310} x2={890} y2={390} color={BLUE} localFrame={localFrame} />
    <Arrow x1={730} y1={530} x2={890} y2={450} color={PURPLE} localFrame={localFrame} />
    <Arrow x1={1200} y1={390} x2={1495} y2={310} color={BLUE} delay={20} localFrame={localFrame} />
    <Arrow x1={1200} y1={450} x2={1495} y2={530} color={PURPLE} delay={20} localFrame={localFrame} />
  </SceneShell>
);

const DepthConditionScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Label style={{ position: "absolute", left: 130, top: 150, color: INK, fontSize: 48 }}>
      Depth-conditioned Generation: T + Depth → RGB
    </Label>
    <Box color={PURPLE} fill={LIGHT_PURPLE} style={{ position: "absolute", left: 180, top: 430, width: 280 }}>
      Depth Condition
      <div style={{ fontSize: 44, marginTop: 18 }}>lock</div>
    </Box>
    <Box color={BLUE} fill="#eff6ff" style={{ position: "absolute", left: 180, top: 245, width: 280 }}>
      RGB Noise
    </Box>
    <Box color={GREEN} fill="#dcfce7" style={{ position: "absolute", left: 720, top: 335, width: 410 }}>
      JointConn-v2 Geometry Guidance
    </Box>
    <Box color={BLUE} fill={LIGHT_BLUE} style={{ position: "absolute", right: 210, top: 335, width: 330 }}>
      Generated RGB
    </Box>
    <Arrow x1={460} y1={290} x2={720} y2={370} color={BLUE} localFrame={localFrame} />
    <Arrow x1={460} y1={485} x2={720} y2={430} color={PURPLE} localFrame={localFrame} />
    <Arrow x1={1130} y1={395} x2={1380} y2={395} color={BLUE} delay={18} localFrame={localFrame} />
    <div
      style={{
        ...baseFont,
        position: "absolute",
        left: 235,
        top: 665,
        color: PURPLE,
        fontSize: 28,
        fontWeight: 800,
      }}
    >
      λy = 0 · my = 0 · ty = 0
    </div>
  </SceneShell>
);

const SummaryScene = ({ localFrame }: { localFrame: number }) => (
  <SceneShell localFrame={localFrame}>
    <Img
      src={staticFile("jointconn_v2_framework.png")}
      style={{
        position: "absolute",
        left: 128,
        top: 120,
        width: 780,
        height: 570,
        objectFit: "contain",
        background: "#fff",
        borderRadius: 8,
        border: "1px solid #dbe3ef",
        boxShadow: "0 22px 58px rgba(15,23,42,0.14)",
      }}
    />
    <div style={{ position: "absolute", right: 130, top: 190, display: "grid", gap: 34 }}>
      {[
        ["Selective Cross-modal Communication", BLUE],
        ["Geometry-aware Edge Guidance", YELLOW],
        ["Stable Weighted Flow Matching", GREEN],
      ].map(([text, color], index) => (
        <Box
          key={text}
          color={color}
          fill={color === YELLOW ? "#fef9c3" : color === BLUE ? LIGHT_BLUE : "#dcfce7"}
          style={{
            width: 650,
            opacity: interpolate(localFrame, [index * 28, index * 28 + 24], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
            transform: `translateX(${interpolate(localFrame, [index * 28, index * 28 + 24], [40, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: ease,
            })}px)`,
          }}
        >
          {text}
        </Box>
      ))}
    </div>
    <Label style={{ position: "absolute", right: 226, top: 655, color: INK, fontSize: 46 }}>
      JointConn-v2 with GCM-WFM
    </Label>
  </SceneShell>
);

const renderScene = (sceneIndex: number, localFrame: number) => {
  switch (sceneIndex) {
    case 0:
      return <TitleScene localFrame={localFrame} />;
    case 1:
      return <TasksScene localFrame={localFrame} />;
    case 2:
      return <MotivationScene localFrame={localFrame} />;
    case 3:
      return <CoreIdeaScene localFrame={localFrame} />;
    case 4:
      return <TokenizationScene localFrame={localFrame} />;
    case 5:
      return <ControllerScene localFrame={localFrame} />;
    case 6:
      return <BackboneScene localFrame={localFrame} />;
    case 7:
      return <ConnectorSitesScene localFrame={localFrame} />;
    case 8:
      return <SwapQScene localFrame={localFrame} />;
    case 9:
      return <GeometryBiasScene localFrame={localFrame} />;
    case 10:
      return <ContentGateScene localFrame={localFrame} />;
    case 11:
      return <RoutingScene localFrame={localFrame} />;
    case 12:
      return <ResidualFusionScene localFrame={localFrame} />;
    case 13:
      return <LossScene localFrame={localFrame} />;
    case 14:
      return <AlphaScene localFrame={localFrame} />;
    case 15:
      return <JointInferenceScene localFrame={localFrame} />;
    case 16:
      return <DepthConditionScene localFrame={localFrame} />;
    default:
      return <SummaryScene localFrame={localFrame} />;
  }
};

export const JointConnVideo = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const sceneFrames = SCENE_SECONDS * fps;
  const sceneIndex = Math.min(SCENE_COUNT - 1, Math.floor(frame / sceneFrames));
  const localFrame = frame - sceneIndex * sceneFrames;

  return (
    <AbsoluteFill style={baseFont}>
      <Background />
      <Header sceneIndex={sceneIndex} />
      {renderScene(sceneIndex, localFrame)}
      <CaptionBar />
    </AbsoluteFill>
  );
};
