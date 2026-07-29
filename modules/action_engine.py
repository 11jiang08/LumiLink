# modules/action_engine.py
"""
线下破冰与行动指南生成模块 (Action Engine)
核心功能：
1. 生成 3 种不同风格的线上破冰开场白 (直球型/幽默型/社恐专属型)。
2. 生成包含时间、地点、互动建议与避坑指南的《专属线下破冰行动指南》。
"""

import json
import logging
from openai import OpenAI
import config
from mock_db import get_user_by_id

logger = logging.getLogger(__name__)

# 初始化 OpenAI / DeepSeek 客户端
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)

ACTION_ENGINE_PROMPT = """
你是一个高情商的校园社交顾问与活动规划 Agent。
请根据【当前用户 User A】和【匹配用户 User B】的档案、性格（包含缺点与避雷点）以及当下的【场景/意图】，生成一份破冰行动方案。

【生成要求】：
1. 三种风格开场白：
   - 直球型：直接表明共同兴趣或意图，高效明快。
   - 幽默型：用轻松搞笑、梗感或自嘲拉近距离。
   - 社恐专属型：极低心理负担、给对方留有退路、温暖无压力的开口方式。
2. 线下行动指南：
   - 场所建议：必须结合双方性格（例如如果双方社恐或不喜欢吵闹，推荐猫咖、安静的操场看台或咖啡馆，避开人多嘈杂处）。
   - 互动方案：围绕共同兴趣或意图设计低门槛互动（如：戴耳机一起跑圈、互相监督刷题、一起去吃小火锅）。
   - 避坑提示：结合 User B 的【雷点/禁忌】，提醒 User A 聊天时需要避开的敏感话题或行为。

【输出格式要求】：
请严格输出符合以下结构的 JSON 格式数据，不要包含任何 markdown 标记（如 ```json ... ```）：
{
    "icebreakers": {
        "direct": "直球型开场白文本",
        "humorous": "幽默型开场白文本",
        "introvert_friendly": "社恐专属型开场白文本"
    },
    "offline_guide": {
        "location_recommendation": "推荐见面地点及理由",
        "activity_idea": "具体的活动与互动方案",
        "avoid_pitfalls": "避坑提示（例如：对方不喜欢被探究过多隐私，建议聊共同喜欢的科幻/跑圈即可）"
    }
}
"""

def generate_icebreaker_and_guide(user_a_profile: dict, match_result: dict) -> dict:
    """
    生成破冰开场白与线下行动指南
    
    :param user_a_profile: 当前用户 Profile
    :param match_result: llm_matcher 返回的匹配结果
    :return: 包含破冰开场白与行动指南的字典
    """
    matched_user_id = match_result.get("best_match_id")
    user_b_profile = get_user_by_id(matched_user_id)
    
    matched_user_id = match_result.get("best_match_id")
    user_b_obj = get_user_by_id(matched_user_id)
    
    # 如果查到了 UserProfile 对象，转成 dict；否则使用默认保底 dict
    if user_b_obj and hasattr(user_b_obj, "to_dict"):
        user_b_profile = user_b_obj.to_dict()
    elif isinstance(user_b_obj, dict):
        user_b_profile = user_b_obj
    else:
        user_b_profile = {
            "nickname": match_result.get("best_match_name", "校友"),
            "hobbies": ["校园生活", "自习"],
            "weaknesses": ["社恐"],
            "landmines": ["大吵大闹"],
            "cv_scene": "校园"
        }

    prompt_payload = {
        "user_a": user_a_profile,
        "user_b": user_b_profile,
        "match_reason": match_result.get("match_reason", "")
    }

    user_prompt = f"请为以下匹配成功的两位用户生成破冰方案：\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": ACTION_ENGINE_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()

        # 清理可能存在的 markdown 代码块标记
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        result = json.loads(content)
        return result

    except Exception as e:
        logger.error(f"Action Engine 调用失败，启用 Fallback 本地保底逻辑: {e}")
        return _fallback_guide(user_a_profile, user_b_profile)


def _fallback_guide(user_a: dict, user_b: dict) -> dict:
    """本地保底降级逻辑，防止现场演示超时或报错"""
    b_name = user_b.get("name", "对方")
    scene = user_a.get("current_scene", "操场/图书馆")
    
    return {
        "icebreakers": {
            "direct": f"嗨！看到系统匹配到你也在【{scene}】附近，今晚要不要一起？",
            "humorous": f"系统说我们俩配对成功率很棒，看来是时候组个【{scene}】搭子小分队了！",
            "introvert_friendly": f"你好呀！我也刚好在【{scene}】附近，如果不打扰的话，可以搭个组~ 如果你今天不太方便也没关系的！"
        },
        "offline_guide": {
            "location_recommendation": f"建议预约在【{scene}】附近较安静的休息区或看台。",
            "activity_idea": "低压互动：无需过多言语交流，先共同进行当前的即时意图活动（如共同刷题或跑步）。",
            "avoid_pitfalls": f"提示：{b_name} 更喜欢自然轻松的氛围，避免打探过多个人私密话题，从共同兴趣聊起最佳。"
        }
    }