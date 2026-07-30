# -*- coding: utf-8 -*-
"""
LumiLink（LL） - 主程序与 Gradio 交互界面

本模块负责构建系统的 Web 前端交互界面，整合多模态感知输入（文本问卷与计算机视觉识别），
调度逆向匹配引擎及 Action Engine，生成最终的匹配结果、破冰策略与线下行动指南。
"""

import logging
import gradio as gr
import config
from pathlib import Path
from user_profile import UserProfile, MatchResult
from modules import fuse_multimodal_inputs, match_user, generate_icebreaker_and_guide
from modules.emotion_analyzer import EmotionAnalyzer
from modules.cv_perception import analyze_image
from mock_db import MOCK_USERS
from modules.map_generator import generate_live_campus_map

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """动态增加兴趣爱好输入框的显示数量。"""
    new_count = min(current_count + 1, MAX_HOBBIES)
    updates = [gr.update(visible=(i < new_count)) for i in range(MAX_HOBBIES)]
    return [new_count] + updates


def on_image_upload(image_input):
    """图片上传/清空时的实时异步回调，调用 CV 模块进行场景与目标识别。"""
    if image_input is None:
        return "🔍 **视觉感知诊断**：未检测到图片"
    
    res = analyze_image(image_input)
    scene = res.get("detected_scene", "未知场景")
    objects = res.get("detected_objects", [])
    
    objects_str = ", ".join(objects) if objects else "未检测到特定特征物品"
    return f"🔍 **视觉感知诊断**：\n- **场景**：【{scene}】\n- **物品**：[{objects_str}]"


def set_button_loading():
    """点击匹配时立即调用的前置函数：改变按钮文字和样式为红粉色动效态"""
    return gr.update(value="⏳ 正在微光逆向匹配中...", elem_classes=["matching-btn"])


def run_pipeline(
    nickname, q1_val, q2_val,
    h1, h2, h3, h4, h5,
    weaknesses, landmines, text_intent, image_input, q7_val
):
    """核心业务流水线：多模态融合 -> 构建用户画像 -> LLM 逆向匹配 -> 生成破冰与行动指南。"""
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


# 全局仪容分析器实例（单例懒加载）
_emotion_analyzer = None


def run_emotion_check(image_np):
    """仪容自检：接收摄像头帧，返回科技感 HTML 仪表盘"""
    global _emotion_analyzer
    if image_np is None:
        return '<div class="emotion-placeholder">📷 点击开启摄像头，开始对镜自检</div>'
    try:
        if _emotion_analyzer is None:
            _emotion_analyzer = EmotionAnalyzer()
        report = _emotion_analyzer.analyze(image_np)
        return report.to_html()
    except Exception as e:
        logger.error(f"仪容自检异常: {e}")
        return f'<div class="emotion-placeholder">⚠️ 仪容自检出错：{e}</div>'


# ---------------------------------------------------------------------------
# 💎 UI 全局视觉重构 CSS (浪漫紫罗兰玻璃拟态 + 浅色防闪烁仪表盘)
# ---------------------------------------------------------------------------
css_path = Path(__file__).parent / "style.css"
custom_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

# ---------------------------------------------------------------------------
# Gradio 布局与交互构建 (UI Layout)
# ---------------------------------------------------------------------------
with gr.Blocks(theme=theme, title="✨ LumiLink") as demo:
    hobby_count_state = gr.State(value=1)

    # ===== 开屏动画 + 主题切换 + 匹配加载动画（js_on_load 注入，避免 <script> 被 sanitize）=====
    gr.HTML(
        value="""
<div id="ll-intro">
  <canvas id="ll-intro-canvas"></canvas>
  <div class="ll-intro-title">LumiLink</div>
  <div class="ll-intro-subtitle">基于真实自我剖析与多模态感知的校园 AI 社交平台 · 点击任意位置进入</div>
</div>
<button id="ll-theme-toggle" title="切换主题">🌙</button>
<div id="ll-loading">
  <canvas id="ll-loading-canvas"></canvas>
  <div class="ll-loading-core">
    <div class="ll-core-dot"></div>
    <div class="ll-core-ring"></div>
    <div class="ll-core-ring"></div>
    <div class="ll-core-ring"></div>
    <div class="ll-core-ring"></div>
  </div>
  <div class="ll-loading-text">微光匹配中</div>
  <div class="ll-loading-sub">正在解析你的社交电池...</div>
  <div class="ll-loading-bar"><div class="ll-loading-bar-fill"></div></div>
  <div class="ll-loading-pct">0%</div>
</div>
""",
        js_on_load="""
setTimeout(function(){
  function initStarfield(canvasId, starCount, connectDist){
    var cv=document.getElementById(canvasId);
    if(!cv) return {stop:function(){}};
    var ctx=cv.getContext('2d');
    var stars=[], animId=null, start=Date.now();
    function resize(){cv.width=window.innerWidth;cv.height=window.innerHeight;}
    resize(); window.addEventListener('resize',resize);
    for(var i=0;i<starCount;i++){
      stars.push({x:Math.random()*cv.width,y:Math.random()*cv.height,
        r:Math.random()*2.0+0.5,opacity:0,delay:Math.random()*2000,
        vx:(Math.random()-0.5)*0.12,vy:(Math.random()-0.5)*0.12,
        tw:Math.random()*6.28});
    }
    var linePairs=[];
    for(var i=0;i<stars.length;i++){
      for(var j=i+1;j<stars.length;j++){
        var dx=stars[i].x-stars[j].x, dy=stars[i].y-stars[j].y;
        var d=Math.sqrt(dx*dx+dy*dy);
        if(d<connectDist*1.2 && Math.random()<0.06){
          linePairs.push([i,j]);
        }
      }
    }
    function animate(){
      var el=Date.now()-start;
      ctx.clearRect(0,0,cv.width,cv.height);
      stars.forEach(function(s){
        if(el>s.delay) s.opacity=Math.min(1,s.opacity+0.02);
        s.x+=s.vx; s.y+=s.vy;
        if(s.x<0)s.x=cv.width; if(s.x>cv.width)s.x=0;
        if(s.y<0)s.y=cv.height; if(s.y>cv.height)s.y=0;
        s.tw+=0.018;
      });
      if(el>600 && linePairs.length>0){
        for(var p=0;p<linePairs.length;p++){
          var si=stars[linePairs[p][0]], sj=stars[linePairs[p][1]];
          if(!si||!sj) continue;
          var dx=si.x-sj.x, dy=si.y-sj.y;
          var d=Math.sqrt(dx*dx+dy*dy);
          if(d<connectDist){
            var a=(1-d/connectDist)*0.25*si.opacity*sj.opacity;
            ctx.strokeStyle='rgba(155,164,196,'+a+')';
            ctx.lineWidth=0.7;
            ctx.beginPath();ctx.moveTo(si.x,si.y);ctx.lineTo(sj.x,sj.y);ctx.stroke();
          }
        }
      }
      stars.forEach(function(s){
        var t=0.65+0.35*Math.sin(s.tw);
        ctx.fillStyle='rgba(220,225,240,'+(s.opacity*t)+')';
        ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,6.28);ctx.fill();
      });
      animId=requestAnimationFrame(animate);
    }
    animate();
    return {stop:function(){if(animId)cancelAnimationFrame(animId);}};
  }
  var intro=document.getElementById('ll-intro');
  if(intro){
    var introField=initStarfield('ll-intro-canvas',70,120);
    intro.addEventListener('click',function(){
      var title=intro.querySelector('.ll-intro-title');
      var subtitle=intro.querySelector('.ll-intro-subtitle');
      if(title) title.classList.add('slide-up');
      if(subtitle) subtitle.style.opacity='0';
      setTimeout(function(){
        intro.classList.add('fade-out');
        setTimeout(function(){intro.remove();introField.stop();},800);
      },600);
    });
  }
  var toggle=document.getElementById('ll-theme-toggle');
  if(toggle){
    var saved=localStorage.getItem('ll-theme')||'dark';
    document.documentElement.setAttribute('data-theme',saved);
    toggle.textContent=saved==='dark'?'☀️':'🌙';
    toggle.addEventListener('click',function(){
      var cur=document.documentElement.getAttribute('data-theme')||'dark';
      var next=cur==='dark'?'light':'dark';
      document.documentElement.setAttribute('data-theme',next);
      localStorage.setItem('ll-theme',next);
      toggle.textContent=next==='dark'?'☀️':'🌙';
    });
  }
  var loadingEl=document.getElementById('ll-loading');
  var loadingField=null;
  var loadingTimer=null, progressTimer=null, msgIdx=0, curPct=0;
  var loadingSteps=[
    {t:'正在解析你的社交电池', s:'读取性格特质与社交能量值...'},
    {t:'扫描兴趣契合度雷达', s:'在校园微光星图中搜索共振频率...'},
    {t:'交叉比对性格雷点', s:'为加密 B 面档案执行排雷校验...'},
    {t:'融合视觉感知场景', s:'ResNet18 正在解析随手拍场景...'},
    {t:'逆向匹配最佳微光搭子', s:'LLM 引擎正在做最后一轮互补推演...'},
    {t:'生成高情商破冰策略', s:'为你的破冰风格定制开场白...'}
  ];
  function setLoadMsg(txt, sub){
    var tEl=loadingEl.querySelector('.ll-loading-text');
    var sEl=loadingEl.querySelector('.ll-loading-sub');
    if(tEl){tEl.style.opacity='0';}
    if(sEl){sEl.style.opacity='0';}
    setTimeout(function(){
      if(tEl){tEl.textContent=txt;tEl.style.opacity='0.95';}
      if(sEl){sEl.textContent=sub;sEl.style.opacity='1';}
    }, 300);
  }
  function setLoadPct(p){
    var bar=loadingEl.querySelector('.ll-loading-bar-fill');
    var pct=loadingEl.querySelector('.ll-loading-pct');
    var v=Math.max(0,Math.min(100,p));
    if(bar){bar.style.width=v+'%';}
    if(pct){pct.textContent=Math.round(v)+'%';}
  }
  function showLoading(){
    if(!loadingEl) return;
    loadingEl.classList.add('active');
    loadingField=initStarfield('ll-loading-canvas',40,100);
    msgIdx=0; curPct=4;
    setLoadMsg(loadingSteps[0].t, loadingSteps[0].s);
    setLoadPct(curPct);
    var stepCount=loadingSteps.length;
    var stepInterval=2200;  // 每步文案停留时长
    loadingTimer=setInterval(function(){
      msgIdx=(msgIdx+1)%stepCount;
      setLoadMsg(loadingSteps[msgIdx].t, loadingSteps[msgIdx].s);
    }, stepInterval);
    // 进度条：缓慢推进到 92%，等待真实完成
    progressTimer=setInterval(function(){
      if(curPct<92){
        // 越接近 92 越慢，制造"即将完成"的期待感
        var remaining=92-curPct;
        curPct+=Math.max(0.4, remaining*0.06);
        setLoadPct(curPct);
      }
    }, 180);
  }
  function hideLoading(){
    if(!loadingEl) return;
    setLoadPct(100);
    var tEl=loadingEl.querySelector('.ll-loading-text');
    var sEl=loadingEl.querySelector('.ll-loading-sub');
    if(tEl){tEl.textContent='匹配完成'; tEl.style.opacity='0.95';}
    if(sEl){sEl.textContent='微光搭子已找到，正在呈现...'; sEl.style.opacity='1';}
    setTimeout(function(){
      loadingEl.classList.remove('active');
      if(loadingField){loadingField.stop();loadingField=null;}
    }, 600);
    if(loadingTimer){clearInterval(loadingTimer);loadingTimer=null;}
    if(progressTimer){clearInterval(progressTimer);progressTimer=null;}
  }
  function setupObserver(){
    var btn=document.querySelector('#submit-match-btn button')||document.querySelector('#submit-match-btn .normal-btn');
    if(!btn){setTimeout(setupObserver,500);return;}
    var observer=new MutationObserver(function(){
      var txt=btn.textContent||'';
      if(txt.indexOf('匹配中')>=0||btn.className.indexOf('matching')>=0){showLoading();}
      else{hideLoading();}
    });
    observer.observe(btn,{attributes:true,attributeFilter:['class'],childList:true,subtree:true,characterData:true});
  }
  setupObserver();
}, 200);
"""
    )
    
    # 页头 Banner
    gr.HTML(
        """
        <div class="header-banner">
            <h1>✨ LumiLink</h1>
            <p>基于真实自我剖析与多模态感知的校园交友平台 — “社交是需要能量的，我们来测测你的社交电池。”</p>
        </div>
        """
    )
    
    with gr.Row():
        # --- 左侧输入区：表单与问卷 ---
        with gr.Column(scale=6):
            
            # 模块一：社交电池与节奏
            with gr.Column(elem_classes=["custom-card-panel"]):
                gr.HTML('<div class="panel-title">🔋 模块一：社交电池与节奏测试 <span class="badge">基础感知</span></div>')
                
                nickname_in = gr.Textbox(label="你的昵称/代号", value="小明", placeholder="怎么称呼你？")
                
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

            # 模块二：B面真实档案 (加密区)
            with gr.Column(elem_classes=["custom-card-panel"]):
                gr.HTML('<div class="panel-title">🔒 模块二：B面真实档案 <span class="badge">加密避雷</span></div>')
                gr.Markdown("> *“在这里，我们不需要虚假的人设。请告诉我们你最真实的‘B面’，这些信息是保密的，只有算法能读懂。”*")
                
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

            # 模块三：偏好与即时状态
            with gr.Column(elem_classes=["custom-card-panel"]):
                gr.HTML('<div class="panel-title">📸 模块三：偏好与即时多模态状态</div>')
                
                intent_in = gr.Textbox(
                    label="Q5 (兴趣爱好)：为了给你寻找志同道合的伙伴，请在下方填入你的兴趣爱好", 
                    value="", 
                )
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
                    type="pil"
                )
                
                cv_result_out = gr.Markdown(
                    value="🔍 **视觉感知诊断**：未检测到图片",
                    elem_classes=["result-card-placeholder"]
                )

                q7_in = gr.Radio(
                    choices=list(Q7_MAP.keys()),
                    value="C. “温和点，简单打个招呼，顺其自然。”",
                    label="Q7 (破冰风格)：匹配成功后，你希望我们以什么方式帮你开启第一句话？"
                )
                
                submit_btn = gr.Button("🚀 开启微光逆向匹配", variant="primary", size="lg", elem_classes=["normal-btn"], elem_id="submit-match-btn")

        # --- 右侧展示区：算法匹配结果 & 模块四 ---
        with gr.Column(scale=6):
            
            # 匹配结果看板
            with gr.Column(elem_classes=["custom-card-panel"]):
                gr.HTML('<div class="panel-title">🎉 匹配结果与分析</div>')
                
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

            # 模块四：准备见面（实时仪容自检）
            with gr.Column(elem_classes=["custom-card-panel"]):
                gr.HTML('<div class="panel-title">🪞 准备见面（实时仪容自检） <span class="badge">MediaPipe AI</span></div>')
                gr.Markdown("> 开启摄像头后，AI 会**实时**分析你的微笑度、紧张度、活力度与自信度，助你以最佳状态赴约。")
                
                emotion_cam = gr.Image(
                    label="点击开启摄像头，对镜自检",
                    sources=["webcam"],
                    type="numpy",
                    streaming=True,
                )
                
                emotion_btn = gr.Button("📷 截图分析当前画面", elem_classes=["normal-btn"])
                
                emotion_out = gr.HTML(
                    value='<div class="emotion-placeholder">⏳ 等待摄像头开启<br/>开启后 AI 将实时分析你的微笑度、紧张度与活力自信</div>'
                )

            # 🌟 新增：右下侧 - 校园微光分布地图 🌟
            with gr.Column(elem_classes=["custom-card-panel"]):
                gr.HTML(
                    '<div class="panel-title">🌐 实时校园微光星图 '
                    '<span class="badge" style="background: linear-gradient(135deg, #a855f7, #ec4899); color: white;">'
                    '实时等待中</span></div>'
                )
                gr.Markdown("> *地图上闪烁的每个紫罗兰微光，都是此时此刻正寻找真实契合搭子的校友。*")
                
                # 初始化地图 HTML
                initial_map_html = generate_live_campus_map(MOCK_USERS, user_count=16)
                
                campus_map_out = gr.HTML(value=initial_map_html)

    # ---------------------------------------------------------------------------
    # 交互事件绑定 (Event Bindings)
    # ---------------------------------------------------------------------------
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
    
    # 3. 点击提交触发异步链式调用
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

    # 4. 摄像头流式分析 (平滑刷新间隔 0.5s，降低抖动闪烁)
    emotion_cam.stream(
        fn=run_emotion_check,
        inputs=[emotion_cam],
        outputs=[emotion_out],
        time_limit=30,
        stream_every=0.5,
    )

    # 5. 手动截图分析按钮
    emotion_btn.click(
        fn=run_emotion_check,
        inputs=[emotion_cam],
        outputs=[emotion_out],
    )

app = demo

if __name__ == "__main__":
    demo.queue().launch(share=False, css=custom_css)