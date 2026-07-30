# -*- coding: utf-8 -*-
"""
地图生成模块 (Map Generator)
用于生成基于上海交通大学（闵行校区）的 2030 科幻紫罗兰/蓝光呼吸灯校园用户分布图。
"""

import random
import folium

# 上海交通大学（闵行校区）中心地理坐标 (纬度, 经度)
SJTU_MINHANG_CENTER = [31.0253, 121.4370]  


def generate_live_campus_map(mock_users: list, user_count: int = 16) -> str:
    """
    基于交大闵行校区生成带动态闪烁光点的校园地图 HTML 字符串。
    
    :param mock_users: 候选用户列表
    :param user_count: 地图上展示的光点用户数量
    :return: HTML 格式的地图代码字符串
    """
    # 1. 创建基础地图（CartoDB dark_matter 暗夜深色主题）
    m = folium.Map(
        location=SJTU_MINHANG_CENTER,
        zoom_start=15,  # 15级缩放可完整覆盖交大闵行主校区
        tiles="CartoDB dark_matter",
        control_scale=False,
        zoom_control=True
    )

    # 2. 注入 CSS 脉冲呼吸光点动画 + 弹窗深色化（配合 light 模式 invert 滤镜反转成浅色）
    pulse_css = """
    <style>
    @keyframes pulse-glow {
        0% {
            transform: scale(0.8);
            box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.7);
        }
        70% {
            transform: scale(1.25);
            box-shadow: 0 0 0 14px rgba(168, 85, 247, 0);
        }
        100% {
            transform: scale(0.8);
            box-shadow: 0 0 0 0 rgba(168, 85, 247, 0);
        }
    }
    .glowing-dot {
        width: 14px;
        height: 14px;
        background-color: #a855f7;
        border: 2px solid #ffffff;
        border-radius: 50%;
        animation: pulse-glow 2s infinite ease-in-out;
        cursor: pointer;
    }
    .glowing-dot-active {
        background-color: #ec4899 !important;
        animation: pulse-glow 1.2s infinite ease-in-out !important;
    }
    /* 弹窗外框深色化：dark 模式原样深底；light 模式经外层 invert 滤镜反转成浅底 */
    .leaflet-popup-content-wrapper,
    .leaflet-popup-tip {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        box-shadow: 0 3px 14px rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
    }
    .leaflet-popup-content { margin: 8px 12px !important; }
    .leaflet-container a.leaflet-popup-close-button { color: #94a3b8 !important; }
    .leaflet-container .leaflet-control-zoom a {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: rgba(168, 85, 247, 0.3) !important;
    }
    .leaflet-container .leaflet-control-zoom a:hover { background: #334155 !important; }
    .leaflet-container .leaflet-control-attribution {
        background: rgba(30, 41, 59, 0.7) !important;
        color: #94a3b8 !important;
    }
    .leaflet-container .leaflet-control-attribution a { color: #c4b5fd !important; }
    </style>
    """
    m.get_root().html.add_child(folium.Element(pulse_css))

    # 3. 围绕交大闵行校区范围（东西约2km，南北约1.5km）进行坐标偏移采样
    sampled_users = random.sample(mock_users, min(user_count, len(mock_users))) if mock_users else []

    for i, u in enumerate(sampled_users):
        # 经纬度偏移量适配交大闵行校区边界
        lat_offset = random.uniform(-0.007, 0.007)
        lng_offset = random.uniform(-0.009, 0.009)
        pos = [SJTU_MINHANG_CENTER[0] + lat_offset, SJTU_MINHANG_CENTER[1] + lng_offset]

        # 交替使用高亮闪烁粉紫灯
        dot_class = "glowing-dot glowing-dot-active" if i % 3 == 0 else "glowing-dot"

        # 悬浮提示 & 弹窗内容（深底浅字：dark 模式原样显示；light 模式经 invert 滤镜后变浅底深字）
        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; color: #e2e8f0; line-height: 1.5; min-width: 160px; background: #1e293b; padding: 4px 2px;">
            <b style="color: #c4b5fd; font-size: 14px;">✨ {u.nickname}</b><br/>
            <span style="color:#94a3b8;">📍 场景：</span><b style="color:#f1f5f9;">{u.cv_scene or '交大闵行校区中'}</b><br/>
            <span style="color:#94a3b8;">🏷️ 兴趣：</span><span style="color:#e2e8f0;">{", ".join(u.hobbies[:2])}</span><br/>
            <span style="color: #86efac; font-size: 11px; font-weight: 600;">🟢 微光等待匹配中</span>
        </div>
        """

        icon = folium.DivIcon(
            html=f'<div class="{dot_class}"></div>',
            icon_size=(16, 16),
            icon_anchor=(8, 8)
        )

        folium.Marker(
            location=pos,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{u.nickname}（交大闵行校区 · {u.cv_scene or '等待中'}）",
            icon=icon
        ).add_to(m)

    # 4. 包装为带科技感圆角边框的 HTML（边框/阴影交给 style.css 的 .ll-map-wrap 按主题控制）
    map_html = m._repr_html_()
    return f"""
    <div class="ll-map-wrap">
        {map_html}
    </div>
    """