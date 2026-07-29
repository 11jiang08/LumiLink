"""
LumiLink —— 2030 微光相遇 (Lumina Campus Link)
================================================
Gradio 极速交互原型主入口。

完整闭环：拍照输入 → CV 自动填充标签 → 填写问卷 → 点击匹配 → 生成破冰建议。

运行：
    python app.py
然后浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr

from config import APP, CV
from modules.cv_perception import CVPerceiver
from modules.matching_engine import MatchingEngine
from modules.action_engine import ActionEngine
from modules.user_profile import UserProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LumiLink.app")

# ---------- 引擎初始化 ----------
cv_perceiver = CVPerceiver()
matching_engine = MatchingEngine()
action_engine = ActionEngine()


# =========================================================
# 业务回调函数
# =========================================================
def on_perceive(image_path: str | None) -> tuple[str, str]:
    """步骤一回调：视觉感知，自动补全场景与物品标签。"""
    if not image_path:
        return "请先上传或拍照一张照片", ""
    try:
        result = cv_perceiver.perceive(image_path)
        scene = result.get("current_scene", "未知")
        objects = result.get("detected_objects", [])
        tags = ", ".join(objects) if objects else "（未识别到明显物品）"
        summary = (
            f"**当前场景**：{scene}\n\n"
            f"**识别到的物品/兴趣标签**：{tags}\n\n"
            f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"
        )
        return summary, json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.exception("CV 感知失败")
        return f"⚠️ 视觉感知出错：{e}", ""


def on_match(
    nickname: str,
    hobbies: str,
    landmines: str,
    weaknesses: str,
    cv_json: str,
) -> str:
    """步骤二回调：LLM 逆向匹配。"""
    if not (nickname and weaknesses):
        return "⚠️ 请至少填写昵称与个人缺点，才能进行逆向匹配。"

    try:
        cv_info = json.loads(cv_json) if cv_json else {}
    except json.JSONDecodeError:
        cv_info = {}

    user = UserProfile(
        nickname=nickname.strip(),
        hobbies=[h.strip() for h in hobbies.split(",") if h.strip()],
        landmines=[l.strip() for l in landmines.split(",") if l.strip()],
        weaknesses=[w.strip() for w in weaknesses.split(",") if w.strip()],
        cv_scene=cv_info.get("current_scene", ""),
        cv_objects=cv_info.get("detected_objects", []),
    )

    try:
        result = matching_engine.match(user)
        return result.to_markdown()
    except Exception as e:
        logger.exception("匹配失败")
        return f"⚠️ 匹配失败：{e}"


def on_generate_icebreaker(match_result_text: str) -> str:
    """步骤三回调：生成破冰开场白与线下行动指南。"""
    if not match_result_text or "⚠️" in match_result_text:
        return "⚠️ 请先完成匹配，再生成破冰建议。"
    try:
        guide = action_engine.generate(match_result_text)
        return guide
    except Exception as e:
        logger.exception("破冰生成失败")
        return f"⚠️ 破冰建议生成失败：{e}"


# =========================================================
# Gradio UI
# =========================================================
def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP.app_title) as demo:
        gr.Markdown(
            f"# {APP.app_title}\n\n"
            f"> {APP.app_desc}\n\n"
            f"> **核心理念**：在这里，你的缺点不仅安全，还能帮你找到最合拍的搭子。"
        )

        # ---------- 步骤一：视觉感知 ----------
        with gr.Tab("① 视觉感知 (拍照即表达)"):
            gr.Markdown(
                "### 拍/传一张当下的照片，系统自动识别场景与兴趣物品\n"
                "使用 **ResNet18** 做场景分类 + **YOLOv8** 做物品检测。"
            )
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(
                        label="随手拍一张",
                        type="filepath",
                        sources=["upload", "webcam"],
                    )
                    perceive_btn = gr.Button("🔍 自动识别场景与物品", variant="primary")
                with gr.Column():
                    perceive_output = gr.Markdown(label="感知结果")
                    cv_state = gr.Textbox(visible=False)  # 隐藏的 JSON 状态
            perceive_btn.click(
                on_perceive, inputs=[img_input], outputs=[perceive_output, cv_state]
            )

        # ---------- 步骤二：自我剖析问卷 + 匹配 ----------
        with gr.Tab("② 加密缺点问卷 + 逆向匹配"):
            gr.Markdown(
                "### 填写真实自我问卷（缺点只对 AI 可见）\n"
                "AI 会检查「A 的缺点是否命中 B 的雷点」，并寻找能互补你的人。"
            )
            with gr.Row():
                with gr.Column():
                    nickname = gr.Textbox(label="昵称", placeholder="例如：小林")
                    hobbies = gr.Textbox(
                        label="基础兴趣（逗号分隔）",
                        placeholder="科幻小说, 羽毛球, 民谣",
                    )
                    landmines = gr.Textbox(
                        label="性格雷点（绝对不能忍受的特质，逗号分隔）",
                        placeholder="话痨, 不守时, 爱打断别人",
                    )
                    weaknesses = gr.Textbox(
                        label="个人缺点（加密区，仅 AI 可见，逗号分隔）",
                        placeholder="不善言辞, 拖延症, 重度颜控",
                    )
                    match_btn = gr.Button("💞 开始逆向匹配", variant="primary")
                with gr.Column():
                    match_output = gr.Markdown(label="匹配结果")

            match_btn.click(
                on_match,
                inputs=[nickname, hobbies, landmines, weaknesses, cv_state],
                outputs=[match_output],
            )

        # ---------- 步骤三：破冰与线下行动 ----------
        with gr.Tab("③ AI 破冰与线下行动指南"):
            gr.Markdown(
                "### 基于匹配结果生成《专属线下破冰行动指南》\n"
                "包含 3 种风格破冰开场白 + 破冰场所建议 + 话题避坑指南。"
            )
            ice_btn = gr.Button("✨ 生成破冰指南", variant="primary")
            ice_output = gr.Markdown(label="破冰行动指南")
            ice_btn.click(on_generate_icebreaker, inputs=[match_output], outputs=[ice_output])

        gr.Markdown("---\n*视觉感知 → LLM 认知匹配 → 行动引擎* · LumiLink © 2030")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_port=APP.server_port,
        share=APP.share,
        theme=gr.themes.Soft(),
    )
