# app.py
"""
2030“微光相遇” (Lumina Campus Link) - 主程序与 Gradio 交互界面
已调整问题顺序并重新编号
"""

import gradio as gr
import config
from user_profile import UserProfile, MatchResult
from modules import fuse_multimodal_inputs, match_user, generate_icebreaker_and_guide

# Q1 & Q2 选项到性格标签的自动映射
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


def run_pipeline(
    nickname, q1_val, q2_val, hobbies_text, weaknesses, landmines, text_intent, image_input, q7_val
):
    """处理重排顺序后的问卷并运行匹配流程"""
    # 1. 解析基础兴趣
    hobbies_list = [h.strip() for h in hobbies_text.replace("，", ",").split(",") if h.strip()]
    
    # 2. 解析 Q1/Q2 为正向性格特质标签
    personality_traits = [
        Q1_MAP.get(q1_val, "平稳社交"),
        Q2_MAP.get(q2_val, "随和体贴")
    ]
    
    # 3. 多模态视觉感知融合
    multimodal_res = fuse_multimodal_inputs(text_intent, hobbies_list, image_input)
    
    # 4. 构建规范化的 UserProfile 对象
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
    
    # 校验基础必填项
    if not user_profile.is_valid:
        return (
            "⚠️ **请填写昵称并勾选至少一个个人缺点**（即使是加密区，AI 也需要了解真实的你才能做避雷与互补判定哦！）",
            "",
            ""
        )
        
    # 5. 调用 LLM 逆向匹配
    raw_match = match_user(user_profile.to_dict())
    
    # 构建规范的 MatchResult 对象
    match_result_obj = MatchResult(
        matched_user_id=raw_match.get("best_match_id", ""),
        matched_nickname=raw_match.get("best_match_name", "精选搭子"),
        compatibility_score=raw_match.get("compatibility_score", 88) / 100.0,
        avoid_mine_success=0.95 if raw_match.get("avoid_taboo_success", True) else 0.60,
        reason=raw_match.get("match_reason", ""),
        hidden_analysis=f"当前用户加密缺点：{user_profile.weaknesses}\n性格特质：{user_profile.personality}\n命中排雷：已避开对方雷点！\n互补亮点：{raw_match.get('complementary_highlights', [])}"
    )
    
    # 6. 调用 Action Engine 生成破冰与行动建议
    action_res = generate_icebreaker_and_guide(user_profile.to_dict(), raw_match)
    
    # 7. 格式化输出（高亮用户在 Q7 选择的偏好开场白风格）
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
    
    return match_md, icebreaker_md, guide_md


# ==========================================
# Gradio Blocks 界面搭建（重新编号版）
# ==========================================
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(theme=theme, title="2030 微光相遇 (Lumina Campus Link)") as demo:
    gr.Markdown(
        """
        # ✨ 2030“微光相遇” (Lumina Campus Link)
        ### 基于真实自我剖析与多模态感知的校园交友平台
        > **“社交是需要能量的，我们来测测你的‘社交电池’。”**
        """
    )
    
    with gr.Row():
        # 左侧：新版问卷输入区
        with gr.Column(scale=5):
            nickname_in = gr.Textbox(label="你的昵称/代号", value="小明", placeholder="怎么称呼你？")
            
            # --- 模块一：社交电池与节奏测试 ---
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

            # --- 模块二：保密加密区 ---
            gr.Markdown(
                """
                #### 🔒 模块二：B面真实档案（加密保密区）
                > *“在这里，我们不需要虚假的人设。请告诉我们你最真实的‘B面’，这些信息是保密的，只有算法能读懂。”*
                """
            )
            q3_in = gr.CheckboxGroup(
                choices=["拖延症/经常赶截止时间", "选择困难症", "社恐/不善言辞", "容易焦虑/情绪化", "熟人疯子/生人高冷", "过于直接可能伤人"],
                value=["社恐/不善言辞", "选择困难症"],
                label="Q3 (个人缺点)：人无完人，在搭子相处中，你觉得你最容易让对方‘头大’的地方是什么？"
            )
            q4_in = gr.CheckboxGroup(
                choices=["极其讨厌别人迟到", "排斥过度打探隐私", "讨厌社交大吵大闹", "讨厌负能量爆棚", "讨厌说话不回/冷暴力"],
                value=["讨厌社交大吵大闹"],
                label="Q4 (性格雷点)：在一段关系中，什么行为会让你瞬间想直接‘断联’？"
            )

            # --- 模块三：偏好与即时状态 ---
            gr.Markdown("#### 📸 模块三：偏好与即时状态")
            q5_in = gr.Textbox(
                label="Q5 (兴趣爱好)：为了给你寻找志同道合的伙伴，请在这里填入你的兴趣爱好，用逗号隔开",
                value="硬核科幻, 周杰伦, 跑圈, 食堂火锅",
                placeholder="例如：看书, 摄影, 猫咪, 考研刷题"
            )
            intent_in = gr.Textbox(
                label="Q6 (当下意图/想做什么)：给算法看一眼你此时的状态/当下最想做的事情", 
                value="今晚想去操场跑步，顺便找人聊天", 
                placeholder="例如：想找人一起去图书馆赶作业/去食堂吃火锅"
            )
            q7_in = gr.Radio(
                choices=list(Q7_MAP.keys()),
                value="C. “温和点，简单打个招呼，顺其自然。”",
                label="Q7 (破冰风格)：匹配成功后，你希望我们以什么方式帮你开启第一句话？"
            )
            image_in = gr.Image(
                            label="随手拍：对着你现在的书桌、正在读的课本或者想去的操场拍一张照片", 
                            type="pil"
            )
            
            submit_btn = gr.Button("🚀 开启微光逆向匹配", variant="primary", size="lg")

        # 右侧：结果展示区
        with gr.Column(scale=5):
            match_out = gr.Markdown("### 💞 匹配卡片（等待填写问卷...）")
            icebreaker_out = gr.Markdown("### 💬 破冰开场白建议")
            guide_out = gr.Markdown("### 🗺️ 线下行动指南")

    # 事件绑定
    submit_btn.click(
        fn=run_pipeline,
        inputs=[nickname_in, q1_in, q2_in, q5_in, q3_in, q4_in, intent_in, image_in, q7_in],
        outputs=[match_out, icebreaker_out, guide_out]
    )

app = demo

if __name__ == "__main__":
    demo.launch(share=False)