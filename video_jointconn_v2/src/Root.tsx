import "./index.css";
import { Composition } from "remotion";
import { JointConnVideo } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="JointConnV2MethodVideo"
        component={JointConnVideo}
        durationInFrames={5400}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
