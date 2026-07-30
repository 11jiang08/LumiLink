# modules/emotion_analyzer.py
"""
仪容自检模块 (Emotion & Composure Analyzer)
基于 MediaPipe FaceMesh 478 个面部关键点 + 虹膜追踪，输出三维度评分：
1. 微笑度（嘴角上扬程度 + 嘴宽/脸宽比）
2. 紧张度（眉头间距收紧 + 眉毛下压程度）
3. 眼神稳定度（睁眼程度 EAR + 虹膜居中度）

用于"准备见面"环节：用户对镜自拍 → 3 秒内给出表情管理建议。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    def to_markdown(self) -> str:
        return (
            "### 🪞 仪容自检报告\n\n"
            f"😊 **微笑度 {self.smile_score}%** → {self.smile_feedback}\n\n"
            f"😰 **紧张度 {self.tension_score}%** → {self.tension_feedback}\n\n"
            f"👀 **眼神稳定度 {self.stability_score}%** → {self.stability_feedback}\n\n"
            "---\n"
            f"### 📋 见面时的表情管理建议\n\n{self.overall_advice}"
        )


class EmotionAnalyzer:
    """基于 MediaPipe FaceMesh 的面部表情与仪态分析器。"""

    def __init__(self):
        self._mp_face_mesh = None
        self._face_mesh = None

    def _ensure_initialized(self):
        """懒加载 MediaPipe FaceMesh，避免影响 app 启动速度。"""
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,   # 启用虹膜关键点 (468-477)
                min_detection_confidence=0.5,
            )
            logger.info("MediaPipe FaceMesh 初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe 初始化失败: {e}")
            raise

    def analyze(self, image_np) -> EmotionReport:
        """
        对一张人脸图像做三维度评分。

        :param image_np: numpy ndarray (RGB, HxWx3)，来自 Gradio 的 webcam/upload
        :return: EmotionReport
        """
        self._ensure_initialized()

        if image_np is None:
            return self._fail_report("未接收到图像数据")

        rgb = image_np
        h, w = rgb.shape[:2]
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return self._fail_report("未检测到人脸，请正对镜头、保证光线充足后重试")

        lm = results.multi_face_landmarks[0].landmark

        def pt(idx):
            return (lm[idx].x * w, lm[idx].y * h)

        # 脸宽（左右脸颊 234/454）
        cheek_l = pt(234)
        cheek_r = pt(454)
        face_width = ((cheek_r[0] - cheek_l[0]) ** 2 + (cheek_r[1] - cheek_l[1]) ** 2) ** 0.5 or 1.0

        # ---- 1. 微笑度 ----
        mouth_corner_l = pt(61)
        mouth_corner_r = pt(291)
        lip_top = pt(13)
        lip_bottom = pt(14)
        lip_center_y = (lip_top[1] + lip_bottom[1]) / 2
        corner_avg_y = (mouth_corner_l[1] + mouth_corner_r[1]) / 2
        lip_lift = (lip_center_y - corner_avg_y) / face_width   # 越大越上扬
        mouth_width = ((mouth_corner_r[0] - mouth_corner_l[0]) ** 2 +
                       (mouth_corner_r[1] - mouth_corner_l[1]) ** 2) ** 0.5
        mouth_ratio = mouth_width / face_width
        smile = int(lip_lift * 800 + mouth_ratio * 250)
        smile = max(0, min(100, smile))

        # ---- 2. 紧张度 ----
        brow_inner_l = pt(55)
        brow_inner_r = pt(285)
        upper_eyelid_l = pt(159)
        brow_gap = ((brow_inner_r[0] - brow_inner_l[0]) ** 2 +
                    (brow_inner_r[1] - brow_inner_l[1]) ** 2) ** 0.5 / face_width
        brow_drop = (upper_eyelid_l[1] - brow_inner_l[1]) / face_width
        tension_gap = max(0, min(1, (0.11 - brow_gap) / 0.05))
        tension_drop = max(0, min(1, (0.05 - brow_drop) / 0.03))
        tension = int((tension_gap * 0.5 + tension_drop * 0.5) * 100)
        tension = max(0, min(100, tension))

        # ---- 3. 眼神稳定度 ----
        eye_l_outer = pt(33)
        eye_l_inner = pt(133)
        eye_l_top = pt(159)
        eye_l_bottom = pt(145)
        eye_l_w = ((eye_l_inner[0] - eye_l_outer[0]) ** 2 +
                   (eye_l_inner[1] - eye_l_outer[1]) ** 2) ** 0.5 or 1.0
        eye_l_h = ((eye_l_top[0] - eye_l_bottom[0]) ** 2 +
                   (eye_l_top[1] - eye_l_bottom[1]) ** 2) ** 0.5
        ear = eye_l_h / eye_l_w
        iris_l = pt(468)
        eye_l_center_x = (eye_l_outer[0] + eye_l_inner[0]) / 2
        iris_offset = abs(iris_l[0] - eye_l_center_x) / eye_l_w
        ear_score = max(0, min(1, ear / 0.30))
        iris_score = max(0, min(1, 1 - iris_offset / 0.25))
        stability = int((ear_score * 0.6 + iris_score * 0.4) * 100)
        stability = max(0, min(100, stability))

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
        if score >= 70:
            return "状态很好，保持这个笑容！"
        if score >= 40:
            return "嘴角再微微上扬一点就更亲和了。"
        return "建议自然微笑一下，不用夸张，嘴角上扬即可。"

    @staticmethod
    def _tension_feedback(score: int) -> str:
        if score >= 60:
            return "深呼吸，放松眉毛，你看起来有点紧张。"
        if score >= 35:
            return "略微紧张，舒展眉头会显得更放松。"
        return "很放松，状态在线！"

    @staticmethod
    def _stability_feedback(score: int) -> str:
        if score >= 70:
            return "眼神很自然，见面时就这样看着对方。"
        if score >= 40:
            return "眼神略飘，试着把目光落在对方鼻梁三角区。"
        return "眼神躲闪，建议见面时盯着对方双眼之间，减轻对视压力。"

    @staticmethod
    def _overall_advice(smile: int, tension: int, stability: int) -> str:
        tips = []
        if smile < 60:
            tips.append("- 见面时保持自然微笑，不用僵硬——可以提前想一个开心的小事。")
        if tension >= 50:
            tips.append("- 见面前做 3 次深呼吸（4 秒吸、7 秒屏、8 秒呼），能快速降紧张。")
        if stability < 60:
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
