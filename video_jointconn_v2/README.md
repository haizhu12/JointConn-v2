# JointConn-v2 Method Video

This Remotion project renders a 3-minute no-voiceover method overview video for JointConn-v2 with GCM-WFM.

## Files

```text
src/Composition.tsx              Main 18-shot Remotion composition
src/captions.ts                  Hard subtitle timing data
public/captions.json             Caption JSON in Remotion Caption format
public/subtitles.srt             Matching SRT subtitle file
public/jointconn_v2_framework.png
out/jointconn_v2_method_video.mp4
```

## Commands

```powershell
npm i
npm run lint
npm run render
```

Preview in Remotion Studio:

```powershell
npm run dev
```

Composition:

```text
id: JointConnV2MethodVideo
duration: 180 seconds
resolution: 1920x1080
fps: 30
codec: H.264
```
