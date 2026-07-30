# -*- coding: utf-8 -*-
"""
2030“微光相遇” (Lumina Campus Link) - 主程序与 Gradio 交互界面

本模块负责构建系统的 Web 前端交互界面，整合多模态感知输入（文本问卷与计算机视觉识别），
调度逆向匹配引擎及 Action Engine，生成最终的匹配结果、破冰策略与线下行动指南。
"""

import gradio as gr
import config
from user_profile import UserProfile, MatchResult
from modules import fuse_multimodal_inputs, match_user, generate_icebreaker_and_guide
from modules.emotion_analyzer import EmotionAnalyzer
from modules.cv_perception import analyze_image

# ---------------------------------------------------------------------------
# 静态数据映射表 (Mapping Constants)
# ---------------------------------------------------------------------------
Q1_MAP = {
    "A. 组个局，大家一起去吃火锅通宵玩桌游。": "高社交电量/热衷组局",
    "B. 找一两个好朋友看电影或逛街。": "适中社交电量/偏好小聚",
    "C. 一个人在宿舍追剧、看书，享受静谧时光。": "低社交电量/享受独处"
}

Q2_MAP = {
    "A. “我想去哪，什么时候去，我都计划好了，跟着我走就行。”": "主导规划型/节奏清晰",
    "B. “我有大致想法，但更愿意听听对方意见，咱们一起商量。”": "沟通协商型/注重体验",
    "C. “我都行，你定就好，你定哪我就去哪，我听安排。”": "随和跟随型/极高包容度"
}

Q7_MAP = {
    "A. “直接点，聊共同爱好，别整虚的。”": "直球型",
    "B. “幽默点，用个梗或者搞笑开场，缓解气氛。”": "幽默型",
    "C. “温和点，简单打个招呼，顺其自然。”": "社恐专属型"
}

MAX_HOBBIES = 5  # 动态扩展的兴趣爱好输入框上限


# ---------------------------------------------------------------------------
# 前端回调与状态处理函数 (Event Handlers)
# ---------------------------------------------------------------------------
def add_hobby_input(current_count: int):
    """
    动态增加兴趣爱好输入框的显示数量。

    Args:
        current_count (int): 当前已显示的输入框数量。

    Returns:
        list: 更新后的状态计数以及各输入框的 gr.update 对象列表。
    """
    new_count = min(current_count + 1, MAX_HOBBIES)
    updates = [gr.update(visible=(i < new_count)) for i in range(MAX_HOBBIES)]
    return [new_count] + updates


def on_image_upload(image_input):
    """
    图片上传/清空时的实时异步回调，调用 CV 模块进行场景与目标识别。

    Args:
        image_input (PIL.Image or None): 用户上传的图片。

    Returns:
        str: 格式化后的 Markdown 视觉诊断文本。
    """
    if image_input is None:
        return "🔍 **视觉感知诊断**：未检测到图片"
    
    res = analyze_image(image_input)
    scene = res.get("detected_scene", "未知场景")
    objects = res.get("detected_objects", [])
    
    objects_str = ", ".join(objects) if objects else "未检测到特定特征物品"
    return f"🔍 **视觉感知诊断**：\n- **场景**：【{scene}】\n- **物品**：[{objects_str}]"


def set_button_loading():
    """触发匹配时的前置状态更新，切换按钮样式为加载动画。"""
    return gr.update(value="正在微光逆向匹配......", elem_classes=["matching-btn"])


def run_pipeline(
    nickname, q1_val, q2_val,
    h1, h2, h3, h4, h5,
    weaknesses, landmines, text_intent, image_input, q7_val
):
    """
    核心业务流水线：多模态融合 -> 构建用户画像 -> LLM 逆向匹配 -> 生成破冰与行动指南。

    Returns:
        tuple: (匹配卡片Markdown, 破冰建议Markdown, 线下指南Markdown, 恢复状态的按钮Update对象)
    """
    # 1. 数据清洗与兴趣提取
    raw_hobbies = [h1, h2, h3, h4, h5]
    hobbies_list = [h.strip() for h in raw_hobbies if h and h.strip()]
    if not hobbies_list:
        hobbies_list = ["随性聊天", "校园生活"]
    
    # 2. 映射性格特质标签
    personality_traits = [
        Q1_MAP.get(q1_val, "平稳社交"),
        Q2_MAP.get(q2_val, "随和体贴")
    ]
    
    # 3. 融合文本意图与视觉感知特征
    final_intent = text_intent.strip() if text_intent and text_intent.strip() else "今晚想去操场跑步，顺便找人聊天"
    multimodal_res = fuse_multimodal_inputs(final_intent, hobbies_list, image_input)
    
    # 4. 构建领域实体 UserProfile
    user_profile = UserProfile(
        user_id="current_user",
        nickname=nickname or "真实校友",
        hobbies=multimodal_res["final_tags"],
        weaknesses=weaknesses or ["不善言辞"],
        landmines=landmines or [],
        personality=personality_traits,
        cv_scene=multimodal_res["final_scene"],
        cv_objects=multimodal_res["cv_details"]["extracted_tags"]
    )
    
    # 数据合法性校验
    if not user_profile.is_valid:
        btn_update = gr.update(value="🚀 开启微光逆向匹配", elem_classes=["normal-btn"])
        return (
            "⚠️ **请填写昵称并勾选至少一个个人缺点**（即使是加密区，AI 也需要了解真实的你才能做避雷与互补判定哦！）",
            "",
            "",
            btn_update
        )
        
    # 5. 执行 LLM 逆向匹配
    raw_match = match_user(user_profile.to_dict())
    
    match_result_obj = MatchResult(
        matched_user_id=raw_match.get("best_match_id", ""),
        matched_nickname=raw_match.get("best_match_name", "精选搭子"),
        compatibility_score=raw_match.get("compatibility_score", 88) / 100.0,
        avoid_mine_success=0.95 if raw_match.get("avoid_taboo_success", True) else 0.60,
        reason=raw_match.get("match_reason", ""),
        hidden_analysis=(
            f"当前用户加密缺点：{user_profile.weaknesses}\n"
            f"性格特质：{user_profile.personality}\n"
            f"视觉识别感知场景：【{multimodal_res['final_scene']}】\n"
            f"命中排雷：已避开对方雷点！\n"
            f"互补亮点：{raw_match.get('complementary_highlights', [])}"
        )
    )
    
    # 6. 生成对话策略与线下指南
    action_res = generate_icebreaker_and_guide(user_profile.to_dict(), raw_match)
    
    # 7. 格式化输出文本
    match_md = match_result_obj.to_markdown()
    pref_style = Q7_MAP.get(q7_val, "社恐专属型")
    icebreakers = action_res.get("icebreakers", {})
    
    icebreaker_md = (
        f"### 💬 AI 建议的高情商破冰开场白（已为你优先推荐【{pref_style}】）\n"
        f"- **【直球型】**：`{icebreakers.get('direct', '')}`\n"
        f"- **【幽默型】**：`{icebreakers.get('humorous', '')}`\n"
        f"- **【社恐专属/温和型】**：`{icebreakers.get('introvert_friendly', '')}`"
    )
    
    guide = action_res.get("offline_guide", {})
    guide_md = (
        "### 🗺️ 专属线下破冰行动指南\n"
        f"📍 **推荐见面场所**：{guide.get('location_recommendation', '')}\n\n"
        f"🎯 **低压互动方案**：{guide.get('activity_idea', '')}\n\n"
        f"💡 **话题避坑指南**：{guide.get('avoid_pitfalls', '')}"
    )
    
    btn_update = gr.update(value="🚀 开启微光逆向匹配", elem_classes=["normal-btn"])
    return match_md, icebreaker_md, guide_md, btn_update


def set_button_loading():
    """点击匹配时立即调用的前置函数：改变按钮文字和样式为红色动效态"""
    return gr.update(value="正在微光逆向匹配......", elem_classes=["matching-btn"])


# 全局仪容分析器实例（懒加载，避免影响 app 启动）
_emotion_analyzer = None


def run_emotion_check(image_np):
    """仪容自检：接收摄像头帧，返回科技感 HTML 仪表盘"""
    global _emotion_analyzer
    if image_np is None:
        return '<div class="emotion-placeholder">⚠️ 请先开启摄像头，对镜自检</div>'
    try:
        if _emotion_analyzer is None:
            _emotion_analyzer = EmotionAnalyzer()
        report = _emotion_analyzer.analyze(image_np)
        return report.to_html()
    except Exception as e:
        return f'<div class="emotion-placeholder">⚠️ 仪容自检出错：{e}</div>'


def set_emotion_loading():
    """仪容自检按钮 loading 态"""
    return gr.update(value="正在整理仪容仪表......", elem_classes=["matching-btn"])


# 自定义 CSS 样式
# ---------------------------------------------------------------------------
# UI 样式配置 (Global CSS & Theme)
# ---------------------------------------------------------------------------
custom_css = """
/* 匹配按钮交互动效 */
.normal-btn {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    transition: all 0.4s ease-in-out !important;
}

.matching-btn {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: white !important;
    animation: pulse 1.5s infinite;
    transition: all 0.4s ease-in-out !important;
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* 展示区占位卡片 */
.result-card-placeholder {
    border: 2px dashed #cbd5e1 !important;
    background-color: #f8fafc !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 16px !important;
    min-height: 120px !important;
}

/* ===== 模块四：科技感仪容仪表盘 ===== */
.emotion-dashboard {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
}
.metric-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 14px 12px;
    text-align: center;
    color: #e2e8f0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25);
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(15, 23, 42, 0.4); }
.metric-icon { font-size: 24px; }
.metric-label { font-size: 12px; color: #94a3b8; margin-top: 4px; letter-spacing: 1px; }
.metric-value {
    font-size: 34px;
    font-weight: 800;
    margin: 6px 0;
    font-family: 'Consolas', 'Courier New', monospace;
    line-height: 1;
}
.metric-value .unit { font-size: 16px; opacity: 0.7; }
.metric-card.smile .metric-value { color: #fbbf24; text-shadow: 0 0 14px rgba(251,191,36,0.55); }
.metric-card.tension .metric-value { color: #f87171; text-shadow: 0 0 14px rgba(248,113,113,0.55); }
.metric-card.stability .metric-value { color: #34d399; text-shadow: 0 0 14px rgba(52,211,153,0.55); }
.metric-card.vitality .metric-value { color: #22d3ee; text-shadow: 0 0 14px rgba(34,211,238,0.55); }
.metric-card.confidence .metric-value { color: #a78bfa; text-shadow: 0 0 14px rgba(167,139,250,0.55); }
.metric-bar {
    height: 6px;
    background: #334155;
    border-radius: 3px;
    overflow: hidden;
    margin: 8px 0;
}
.metric-fill {
    height: 100%;
    transition: width 0.35s ease;
    border-radius: 3px;
}
.smile .metric-fill { background: linear-gradient(90deg, #f59e0b, #fbbf24); box-shadow: 0 0 8px rgba(251,191,36,0.6); }
.tension .metric-fill { background: linear-gradient(90deg, #ef4444, #f87171); box-shadow: 0 0 8px rgba(248,113,113,0.6); }
.stability .metric-fill { background: linear-gradient(90deg, #10b981, #34d399); box-shadow: 0 0 8px rgba(52,211,153,0.6); }
.vitality .metric-fill { background: linear-gradient(90deg, #0891b2, #22d3ee); box-shadow: 0 0 8px rgba(34,211,238,0.6); }
.confidence .metric-fill { background: linear-gradient(90deg, #7c3aed, #a78bfa); box-shadow: 0 0 8px rgba(167,139,250,0.6); }
.metric-feedback { font-size: 12px; color: #cbd5e1; margin-top: 6px; line-height: 1.4; }
.advice-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 14px 18px;
    color: #e2e8f0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25);
}
.advice-title { font-size: 14px; font-weight: 700; color: #818cf8; margin-bottom: 8px; }
.advice-body { font-size: 13px; color: #cbd5e1; line-height: 1.6; white-space: pre-line; }
.emotion-placeholder {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px dashed #475569;
    border-radius: 14px;
    padding: 40px 20px;
    text-align: center;
    color: #64748b;
    font-size: 14px;
/* 组件通用卡片封装 */
.custom-card {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-top: 12px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
}

.custom-card p {
    margin: 0 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #334155 !important;
}

.custom-card.image-container,
.custom-card .image-container {
    display: flex !important;
    flex-direction: column !important;
}
"""

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

# ---------------------------------------------------------------------------
# Gradio 布局与交互构建 (UI Layout)
# ---------------------------------------------------------------------------
with gr.Blocks(theme=theme, css=custom_css, title="2030 微光相遇 (Lumina Campus Link)") as demo:
    hobby_count_state = gr.State(value=1)
    
    gr.Markdown(
        """
        # ✨ 2030“微光相遇” (Lumina Campus Link)
        ### 基于真实自我剖析与多模态感知的校园交友平台
        > **“社交是需要能量的，我们来测测你的‘社交电池’。”**
        """
    )
    
    with gr.Row():
        # --- 左侧输入区：表单与问卷 ---
        with gr.Column(scale=5):
            nickname_in = gr.Textbox(label="你的昵称/代号", value="小明", placeholder="怎么称呼你？")
            
            gr.Markdown("#### 🔋 模块一：社交电池与节奏测试")
            q1_in = gr.Radio(
                choices=list(Q1_MAP.keys()),
                value="B. 找一两个好朋友看电影或逛街。",
                label="Q1 (社交电量)：周五晚上，假如没作业，你更倾向于哪种状态？"
            )
            q2_in = gr.Radio(
                choices=list(Q2_MAP.keys()),
                value="B. “我有大致想法，但更愿意听听对方意见，咱们一起商量。”",
                label="Q2 (决策/节奏)：如果临时约朋友去吃饭，你会怎么安排？"
            )

            gr.Markdown(
                """
                #### 🔒 模块二：B面真实档案
                > *“在这里，我们不需要虚假的人设。请告诉我们你最真实的‘B面’，这些信息是保密的，只有算法能读懂。”*
                """
            )
            q3_in = gr.CheckboxGroup(
                choices=config.WEAKNESS_OPTIONS,
                value=["社恐/不善言辞", "选择困难症"],
                label="Q3 (个人缺点)：人无完人，在搭子相处中，你觉得你最容易让对方‘头大’的地方是什么？"
            )
            q4_in = gr.CheckboxGroup(
                choices=config.TABOO_OPTIONS,
                value=["讨厌社交大吵大闹"],
                label="Q4 (性格雷点)：在一段关系中，什么行为会让你瞬间想直接‘断联’？"
            )

            gr.Markdown("#### 📸 模块三：偏好与即时状态")
            
            gr.Markdown("**Q5 (兴趣爱好)：为了给你寻找志同道合的伙伴，请在这里填入你的兴趣爱好**")
            hobby_inputs = []
            hobby_inputs.append(gr.Textbox(label="兴趣 1", value="", placeholder="例如：硬核科幻", visible=True))
            hobby_inputs.append(gr.Textbox(label="兴趣 2", value="", placeholder="例如：夜跑", visible=False))
            hobby_inputs.append(gr.Textbox(label="兴趣 3", value="", placeholder="例如：摄影", visible=False))
            hobby_inputs.append(gr.Textbox(label="兴趣 4", value="", placeholder="例如：猫咪", visible=False))
            hobby_inputs.append(gr.Textbox(label="兴趣 5", value="", placeholder="例如：火锅", visible=False))
            
            add_hobby_btn = gr.Button("➕ 添加兴趣爱好", size="sm")

            intent_in = gr.Textbox(
                label="Q6 (当下意图/想做什么)：给算法看一眼你此时的状态/当下最想做的事情", 
                value="", 
                placeholder="今晚想去操场跑步，顺便找人聊天"
            )
            
            image_in = gr.Image(
                label="📷 随手拍：对着你现在的书桌、正在读的课本或者想去的操场拍一张照片", 
                type="pil",
                elem_classes=["custom-card"]
            )
            
            cv_result_out = gr.Markdown(
                value="🔍 **视觉感知诊断**：未检测到图片",
                elem_classes=["custom-card"]
            )

            q7_in = gr.Radio(
                choices=list(Q7_MAP.keys()),
                value="C. “温和点，简单打个招呼，顺其自然。”",
                label="Q7 (破冰风格)：匹配成功后，你希望我们以什么方式帮你开启第一句话？"
            )
            
            submit_btn = gr.Button("🚀 开启微光逆向匹配", variant="primary", size="lg", elem_classes=["normal-btn"])

        # --- 右侧展示区：算法匹配结果 ---
        with gr.Column(scale=5):
            match_out = gr.Markdown(
                """
                ### 💞 最佳搭子卡片
                > *⏳ 尚无匹配数据*
                > 
                > 请在左侧填写你的“社交电池”问卷并上传随手拍，AI 将为你精准排雷并寻找互补校友。
                """,
                elem_classes=["result-card-placeholder"]
            )
            
            icebreaker_out = gr.Markdown(
                """
                ### 💬 高情商破冰建议
                > *⏳ 等待匹配生成*
                > 
                > 匹配成功后，这里将针对你选定的破冰风格生成专属开场白。
                """,
                elem_classes=["result-card-placeholder"]
            )
            
            guide_out = gr.Markdown(
                """
                ### 🗺️ 专属线下行动指南
                > *⏳ 等待场景解析*
                > 
                > 结合你的即时状态与自训练 ResNet18 感知到的场景，AI 为你制定低压力的见面场所与避坑提示。
                """,
                elem_classes=["result-card-placeholder"]
            )

            # --- 模块四：准备见面（实时仪容自检） ---
            gr.Markdown("---")
            gr.Markdown("#### 🪞 模块四：准备见面（实时仪容自检）")
            gr.Markdown("> 开启摄像头后，AI 会**实时**分析你的微笑度、紧张度、眼神稳定度，随时调整。")
            emotion_cam = gr.Image(
                label="点击开启摄像头，对镜自检",
                sources=["webcam"],
                type="numpy",
                streaming=True,
            )
            emotion_btn = gr.Button("📷 截图分析当前画面", elem_classes=["normal-btn"])
            emotion_out = gr.HTML(
                value='<div class="emotion-placeholder">⏳ 等待摄像头开启<br/>开启后 AI 将实时分析你的微笑度、紧张度与眼神稳定度</div>'
            )

    # --- 事件绑定 ---
    
    # 1. 点击添加兴趣爱好按钮
    add_hobby_btn.click(
        fn=add_hobby_input,
        inputs=[hobby_count_state],
        outputs=[hobby_count_state] + hobby_inputs
    )
    
    # 2. 图像变更触发 CV 感知回调
    image_in.change(
        fn=on_image_upload,
        inputs=[image_in],
        outputs=[cv_result_out]
    )
    
    # 3. 点击提交触发异步链式调用（加载动画 -> 匹配计算）
    submit_btn.click(
        fn=set_button_loading,
        inputs=None,
        outputs=[submit_btn]
    ).then(
        fn=run_pipeline,
        inputs=[
            nickname_in, q1_in, q2_in,
            hobby_inputs[0], hobby_inputs[1], hobby_inputs[2], hobby_inputs[3], hobby_inputs[4],
            q3_in, q4_in, intent_in, image_in, q7_in
        ],
        outputs=[match_out, icebreaker_out, guide_out, submit_btn]
    )

    # 3. 摄像头实时流式分析：开启摄像头后每帧自动触发
    emotion_cam.stream(
        fn=run_emotion_check,
        inputs=[emotion_cam],
        outputs=[emotion_out],
    )

    # 4. 手动分析按钮（上传图片时用）
    emotion_btn.click(
        fn=run_emotion_check,
        inputs=[emotion_cam],
        outputs=[emotion_out],
    )

app = demo

if __name__ == "__main__":
    demo.launch(share=False, theme=theme, css=custom_css)