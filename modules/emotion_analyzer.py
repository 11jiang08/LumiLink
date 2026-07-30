# modules/emotion_analyzer.py
"""
仪容自检模块 (Emotion & Composure Analyzer) - MediaPipe 1.0 Tasks API 版
基于 FaceLandmarker 的 52 个 ARKit blendshapes（表情系数），输出三维度评分：
1. 微笑度（mouthSmileLeft / mouthSmileRight 均值，减去 mouthFrown 抵消误判）
2. 紧张度（browDown 眉下压 + mouthPress 嘴唇紧抿 + mouthShrugLower 咬合）
3. 眼神稳定度（eyeLook* 眼球偏移 + eyeBlink 眨眼程度）

加滑动平均滤波器（默认 5 帧窗口）平滑三维度分数，消除帧间抖动。
"""

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_PATH = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")


@dataclass
class EmotionReport:
    """仪容自检报告。"""
    smile_score: int = 0          # 微笑度 0-100
    tension_score: int = 0        # 紧张度 0-100（越高越紧张）
    stability_score: int = 0      # 眼神稳定度 0-100
    smile_feedback: str = ""
    tension_feedback: str = ""
    stability_feedback: str = ""
    overall_advice: str = ""

    def to_html(self) -> str:
        """科技感仪表盘 HTML（深色卡片 + 发光数值 + 渐变进度条）。"""
        return f"""
<div class="emotion-dashboard">
  <div class="metric-card smile">
    <div class="metric-icon">😊</div>
    <div class="metric-label">微笑度</div>
    <div class="metric-value">{self.smile_score}<span class="unit">%</span></div>
    <div class="metric-bar"><div class="metric-fill" style="width:{self.smile_score}%"></div></div>
    <div class="metric-feedback">{self.smile_feedback}</div>
  </div>
  <div class="metric-card tension">
    <div class="metric-icon">😰</div>
    <div class="metric-label">紧张度</div>
    <div class="metric-value">{self.tension_score}<span class="unit">%</span></div>
    <div class="metric-bar"><div class="metric-fill" style="width:{self.tension_score}%"></div></div>
    <div class="metric-feedback">{self.tension_feedback}</div>
  </div>
  <div class="metric-card stability">
    <div class="metric-icon">👀</div>
    <div class="metric-label">眼神稳定度</div>
    <div class="metric-value">{self.stability_score}<span class="unit">%</span></div>
    <div class="metric-bar"><div class="metric-fill" style="width:{self.stability_score}%"></div></div>
    <div class="metric-feedback">{self.stability_feedback}</div>
  </div>
</div>
<div class="advice-card">
  <div class="advice-title">📋 见面时的表情管理建议</div>
  <div class="advice-body">{self.overall_advice}</div>
</div>
"""

    def to_markdown(self) -> str:
        """Markdown 兼容输出。"""
        return (
            "### 🪞 仪容自检报告（实时）\n\n"
            f"😊 **微笑度 {self.smile_score}%** → {self.smile_feedback}\n\n"
            f"😰 **紧张度 {self.tension_score}%** → {self.tension_feedback}\n\n"
            f"👀 **眼神稳定度 {self.stability_score}%** → {self.stability_feedback}\n\n"
            "---\n"
            f"### 📋 见面时的表情管理建议\n\n{self.overall_advice}"
        )


class EmotionAnalyzer:
    """基于 MediaPipe FaceLandmarker blendshapes 的面部表情分析器，带滑动平均滤波。"""

    def __init__(self, smooth_window: int = 5):
        self._detector = None
        self._smooth_window = smooth_window
        # 三维度历史缓冲，用 deque 自动滚动窗口
        self._smile_hist = deque(maxlen=smooth_window)
        self._tension_hist = deque(maxlen=smooth_window)
        self._stability_hist = deque(maxlen=smooth_window)

    def reset(self):
        """重置滤波器历史（切换用户/重新开始时调用）。"""
        self._smile_hist.clear()
        self._tension_hist.clear()
        self._stability_hist.clear()

    @staticmethod
    def _smooth(value: int, history: deque) -> int:
        """把当前值压入历史队列，返回窗口内平均值。"""
        history.append(value)
        return int(sum(history) / len(history))

    def _ensure_initialized(self):
        """懒加载 FaceLandmarker，避免影响 app 启动速度。"""
        if self._detector is not None:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not Path(_MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"模型文件不存在：{_MODEL_PATH}\n"
                    "请从 https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/face_landmarker.task 下载"
                )

            base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=1,
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker 初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe 初始化失败: {e}")
            raise

    def analyze(self, image_np) -> EmotionReport:
        """
        对一张人脸图像做三维度评分（基于 blendshapes + 滑动平均滤波）。

        :param image_np: numpy ndarray (RGB, HxWx3)，来自 Gradio 的 webcam
        :return: EmotionReport
        """
        self._ensure_initialized()
        import mediapipe as mp

        if image_np is None:
            return self._fail_report("未接收到图像数据")

        rgb = image_np
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return self._fail_report("图像格式异常，需 RGB 三通道")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.face_blendshapes:
            return self._fail_report("未检测到人脸，请正对镜头、保证光线充足后重试")

        # blendshapes 转成字典 {名称: 分数(0-1)}
        bshapes = result.face_blendshapes[0]
        bs = {b.category_name: b.score for b in bshapes}

        # ---- 1. 微笑度：mouthSmile 均值 ----
        smile_raw = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
        smile = int(max(0, smile_raw - frown * 0.3) * 100)
        smile = max(0, min(100, smile))

        # ---- 2. 紧张度：browDown 眉下压 + mouthPress 嘴唇紧抿 ----
        brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
        mouth_press = (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2
        jaw_clench = bs.get("mouthShrugLower", 0)
        tension = int((brow_down * 0.6 + mouth_press * 0.25 + jaw_clench * 0.15) * 100)
        tension = max(0, min(100, tension))

        # ---- 3. 眼神稳定度：眼球偏移越小越稳 ----
        look_in = (bs.get("eyeLookInLeft", 0) + bs.get("eyeLookInRight", 0)) / 2
        look_out = (bs.get("eyeLookOutLeft", 0) + bs.get("eyeLookOutRight", 0)) / 2
        look_up = (bs.get("eyeLookUpLeft", 0) + bs.get("eyeLookUpRight", 0)) / 2
        look_down = (bs.get("eyeLookDownLeft", 0) + bs.get("eyeLookDownRight", 0)) / 2
        gaze_shift = (look_in + look_out + look_up + look_down) / 4
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        stability = int(100 - (gaze_shift * 160 + blink * 40))
        stability = max(0, min(100, stability))

        # ---- 滑动平均滤波：平滑帧间抖动 ----
        smile = self._smooth(smile, self._smile_hist)
        tension = self._smooth(tension, self._tension_hist)
        stability = self._smooth(stability, self._stability_hist)

        return EmotionReport(
            smile_score=smile,
            tension_score=tension,
            stability_score=stability,
            smile_feedback=self._smile_feedback(smile),
            tension_feedback=self._tension_feedback(tension),
            stability_feedback=self._stability_feedback(stability),
            overall_advice=self._overall_advice(smile, tension, stability),
        )

    @staticmethod
    def _smile_feedback(score: int) -> str:
        if score >= 60:
            return "状态很好，保持这个笑容！"
        if score >= 30:
            return "嘴角再微微上扬一点就更亲和了。"
        return "建议自然微笑一下，不用夸张，嘴角上扬即可。"

    @staticmethod
    def _tension_feedback(score: int) -> str:
        if score >= 55:
            return "深呼吸，放松眉毛，你看起来有点紧张。"
        if score >= 30:
            return "略微紧张，舒展眉头会显得更放松。"
        return "很放松，状态在线！"

    @staticmethod
    def _stability_feedback(score: int) -> str:
        if score >= 65:
            return "眼神很自然，见面时就这样看着对方。"
        if score >= 40:
            return "眼神略飘，试着把目光落在对方鼻梁三角区。"
        return "眼神躲闪，建议见面时盯着对方双眼之间，减轻对视压力。"

    @staticmethod
    def _overall_advice(smile: int, tension: int, stability: int) -> str:
        tips = []
        if smile < 50:
            tips.append("- 见面时保持自然微笑，不用僵硬——可以提前想一个开心的小事。")
        if tension >= 45:
            tips.append("- 见面前做 3 次深呼吸（4 秒吸、7 秒屏、8 秒呼），能快速降紧张。")
        if stability < 55:
            tips.append("- 眼神不知道往哪放时，看对方鼻梁三角区，既不躲闪也不冒犯。")
        if not tips:
            tips.append("- 整体状态很棒！保持这份松弛感，做真实的自己就好。")
        return "\n".join(tips)

    @staticmethod
    def _fail_report(msg: str) -> EmotionReport:
        return EmotionReport(
            smile_feedback=msg,
            tension_feedback="—",
            stability_feedback="—",
            overall_advice="请重新拍照后再试。",
        )
