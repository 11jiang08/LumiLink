# modules/emotion_analyzer.py
"""
仪容自检模块 (Emotion & Composure Analyzer) - MediaPipe 1.0 Tasks API 版
基于 FaceLandmarker 的 ARKit blendshapes（表情系数），输出四维度评分：
1. 微笑度（mouthSmile 均值，减去 mouthFrown 抵消误判）
2. 紧张度（browDown 眉下压 + mouthPress 嘴唇紧抿 + mouthShrugLower 咬合）
3. 精神饱满度（反疲惫：低眨眼 + 低眯眼 + 低下唇收紧）
4. 自信度（browInnerUp 内眉上扬 + jawForward 下巴前伸 + 低眉下压）

内置滑动平均滤波器（默认 5 帧窗口）平滑四维度分数，消除帧间抖动。
"""

import logging
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 使用 Path 确保跨平台及路径编码兼容
_BASE_DIR = Path(__file__).resolve().parent.parent
_MODEL_PATH = _BASE_DIR / "models" / "face_landmarker.task"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


@dataclass
class EmotionReport:
    """仪容自检报告（四维度）。"""
    smile_score: int = 0          # 微笑度 0-100
    tension_score: int = 0        # 紧张度 0-100（越高越紧张）
    vitality_score: int = 0       # 精神饱满度 0-100（越高越精神）
    confidence_score: int = 0     # 自信度 0-100
    smile_feedback: str = ""
    tension_feedback: str = ""
    vitality_feedback: str = ""
    confidence_feedback: str = ""
    overall_advice: str = ""

    def to_html(self) -> str:
        """微光点仪表盘（四维度）。"""
        return f"""
<div class="emotion-dashboard">
  <div class="metric-card smile">
    <div class="metric-glow"></div>
    <div class="metric-label">微笑度</div>
    <div class="metric-value">{self.smile_score}<span class="unit">%</span></div>
    <div class="metric-bar"><div class="metric-fill" style="width:{self.smile_score}%"></div></div>
    <div class="metric-feedback">{self.smile_feedback}</div>
  </div>
  <div class="metric-card tension">
    <div class="metric-glow"></div>
    <div class="metric-label">紧张度</div>
    <div class="metric-value">{self.tension_score}<span class="unit">%</span></div>
    <div class="metric-bar"><div class="metric-fill" style="width:{self.tension_score}%"></div></div>
    <div class="metric-feedback">{self.tension_feedback}</div>
  </div>
  <div class="metric-card vitality">
    <div class="metric-glow"></div>
    <div class="metric-label">精神饱满度</div>
    <div class="metric-value">{self.vitality_score}<span class="unit">%</span></div>
    <div class="metric-bar"><div class="metric-fill" style="width:{self.vitality_score}%"></div></div>
    <div class="metric-feedback">{self.vitality_feedback}</div>
  </div>
  <div class="metric-card confidence">
    <div class="metric-glow"></div>
    <div class="metric-label">自信度</div>
    <div class="metric-value">{self.confidence_score}<span class="unit">%</span></div>
    <div class="metric-bar"><div class="metric-fill" style="width:{self.confidence_score}%"></div></div>
    <div class="metric-feedback">{self.confidence_feedback}</div>
  </div>
</div>
<div class="advice-card">
  <div class="advice-title">见面时的表情管理建议</div>
  <div class="advice-body">{self.overall_advice}</div>
</div>
"""

    def to_markdown(self) -> str:
        """Markdown 兼容输出。"""
        return (
            "### 🪞 仪容自检报告（实时）\n\n"
            f"😊 **微笑度 {self.smile_score}%** → {self.smile_feedback}\n\n"
            f"😰 **紧张度 {self.tension_score}%** → {self.tension_feedback}\n\n"
            f"⚡ **精神饱满度 {self.vitality_score}%** → {self.vitality_feedback}\n\n"
            f"💪 **自信度 {self.confidence_score}%** → {self.confidence_feedback}\n\n"
            "---\n"
            f"### 📋 见面时的表情管理建议\n\n{self.overall_advice}"
        )


class EmotionAnalyzer:
    """基于 MediaPipe FaceLandmarker blendshapes 的面部表情分析器，带滑动平均滤波。"""

    def __init__(self, smooth_window: int = 5):
        self._detector = None
        self._smooth_window = smooth_window
        # 四维度历史缓冲，用 deque 自动滚动窗口
        self._smile_hist = deque(maxlen=smooth_window)
        self._tension_hist = deque(maxlen=smooth_window)
        self._vitality_hist = deque(maxlen=smooth_window)
        self._confidence_hist = deque(maxlen=smooth_window)

    def reset(self):
        """重置滤波器历史（切换用户/重新开始时调用）。"""
        self._smile_hist.clear()
        self._tension_hist.clear()
        self._vitality_hist.clear()
        self._confidence_hist.clear()

    @staticmethod
    def _smooth(value: int, history: deque) -> int:
        """把当前值压入历史队列，返回窗口内平均值。"""
        history.append(value)
        return int(sum(history) / len(history))

    def _ensure_initialized(self):
        """懒加载 FaceLandmarker，自动检测/下载模型文件。"""
        if self._detector is not None:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            # 如果模型不存在，尝试自动从 CDN 下载
            if not _MODEL_PATH.exists():
                logger.info(f"未检测到模型文件，正在自动下载至: {_MODEL_PATH} ...")
                _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                try:
                    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
                    logger.info("模型文件下载成功！")
                except Exception as dl_err:
                    logger.error(f"模型下载失败: {dl_err}")
                    raise FileNotFoundError(
                        f"模型文件不存在且无法自动下载：{_MODEL_PATH}\n"
                        f"请手动下载放置：{_MODEL_URL}"
                    ) from dl_err

            base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
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
        对一张人脸图像做四维度评分（基于 blendshapes + 滑动平均滤波）。

        :param image_np: numpy ndarray (RGB, HxWx3)，来自 Gradio 的 webcam
        :return: EmotionReport
        """
        self._ensure_initialized()
        import mediapipe as mp

        if image_np is None:
            return self._fail_report("未检测到图片")

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

        # ---- 1. 微笑度：mouthSmile 均值，减去 mouthFrown 抵消误判 ----
        smile_raw = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
        smile = int(max(0, smile_raw - frown * 0.3) * 100)
        smile = max(0, min(100, smile))

        # ---- 2. 紧张度：browDown 眉下压 + mouthPress 嘴唇紧抿 + mouthShrugLower 咬合 ----
        brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
        mouth_press = (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2
        jaw_clench = bs.get("mouthShrugLower", 0)
        tension = int((brow_down * 0.6 + mouth_press * 0.25 + jaw_clench * 0.15) * 100)
        tension = max(0, min(100, tension))

        # ---- 3. 精神饱满度：低眨眼 + 低眯眼 + 低下唇收紧 = 精神 ----
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        eye_squint = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
        vitality = int(100 - (blink * 30 + eye_squint * 35 + jaw_clench * 35))
        vitality = max(0, min(100, vitality))

        # ---- 4. 自信度：内眉上扬 + 下巴前伸 + 低眉下压 ----
        brow_inner_up = bs.get("browInnerUp", 0)
        jaw_forward = bs.get("jawForward", 0)
        confidence = int((brow_inner_up * 0.35 + jaw_forward * 0.30 + (1 - brow_down) * 0.35) * 100)
        confidence = max(0, min(100, confidence))

        # ---- 滑动平均滤波：平滑帧间抖动 ----
        smile = self._smooth(smile, self._smile_hist)
        tension = self._smooth(tension, self._tension_hist)
        vitality = self._smooth(vitality, self._vitality_hist)
        confidence = self._smooth(confidence, self._confidence_hist)

        return EmotionReport(
            smile_score=smile,
            tension_score=tension,
            vitality_score=vitality,
            confidence_score=confidence,
            smile_feedback=self._smile_feedback(smile),
            tension_feedback=self._tension_feedback(tension),
            vitality_feedback=self._vitality_feedback(vitality),
            confidence_feedback=self._confidence_feedback(confidence),
            overall_advice=self._overall_advice(smile, tension, vitality, confidence),
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
    def _vitality_feedback(score: int) -> str:
        if score >= 65:
            return "精神饱满，眼神有光，状态满分！"
        if score >= 40:
            return "精神尚可，见面前去洗把脸会更清醒。"
        return "略显疲惫，建议见面前喝口水、活动一下肩颈。"

    @staticmethod
    def _confidence_feedback(score: int) -> str:
        if score >= 60:
            return "自信从容，这份气场很加分！"
        if score >= 35:
            return "还可以再挺胸抬头一点，更显自信。"
        return "建议挺胸抬头、微微扬起下巴，提升气场。"

    @staticmethod
    def _overall_advice(smile: int, tension: int, vitality: int, confidence: int) -> str:
        tips = []
        if smile < 50:
            tips.append("- 见面时保持自然微笑，不用僵硬——可以提前想一个开心的小事。")
        if tension >= 45:
            tips.append("- 见面前做 3 次深呼吸（4 秒吸、7 秒屏、8 秒呼），能快速降紧张。")
        if vitality < 50:
            tips.append("- 精神不够时，见面前用冷水洗脸或做 10 秒开合跳，快速唤醒状态。")
        if confidence < 40:
            tips.append("- 自信不够时，挺胸抬头、放慢语速，肢体姿态会反过来影响心理状态。")
        if not tips:
            tips.append("- 整体状态很棒！保持这份松弛感，做真实的自己就好。")
        return "\n".join(tips)

    @staticmethod
    def _fail_report(msg: str) -> EmotionReport:
        return EmotionReport(
            smile_feedback=msg,
            tension_feedback="—",
            vitality_feedback="—",
            confidence_feedback="—",
            overall_advice="请重新拍照后再试。",
        )