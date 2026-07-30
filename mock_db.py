# -*- coding: utf-8 -*-
"""
Replace project name 'Lumina Campus Link' with 'LumiLink'
LumiLink（微光相遇） - 模拟用户数据库 (Mock Database)

本模块提供 50 份高质量校友画像数据，涵盖多样化的性格、兴趣、雷点、加密缺点及多模态视觉感知场景，
专为 LLM 逆向匹配引擎提供丰富的候选人匹配池。
"""

from user_profile import UserProfile

MOCK_USERS = [
    # ---------------- 1. 运动与户外类 (user_001 ~ user_009) ----------------
    UserProfile(
        user_id="user_001",
        nickname="林小乐",
        hobbies=["硬核科幻", "夜跑", "周杰伦", "AI 绘画"],
        landmines=["讨厌社交大吵大闹", "讨厌负能量爆棚"],
        weaknesses=["社恐/不善言辞", "熟人疯子/生人高冷"],
        personality=["极佳的倾听者", "专注细心"],
        cv_scene="操场跑道",
        cv_objects=["跑步", "戴耳机"]
    ),
    UserProfile(
        user_id="user_002",
        nickname="陆浩然",
        hobbies=["健身", "硬核科幻", "无人机", "高数"],
        landmines=["极其讨厌别人迟到", "拖延症/经常赶截止时间"],
        weaknesses=["过于理性/不够圆滑"],
        personality=["执行力极强", "极其守时"],
        cv_scene="操场跑道",
        cv_objects=["体测", "跑步"]
    ),
    UserProfile(
        user_id="user_003",
        nickname="王羽飞",
        hobbies=["羽毛球", "网球", "运动摄影", "周杰伦"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["胜负欲太强/容易较真"],
        personality=["阳光有活力", "会带动气氛"],
        cv_scene="体育馆/操场",
        cv_objects=["羽毛球拍", "运动水壶"]
    ),
    UserProfile(
        user_id="user_004",
        nickname="张雷烈",
        hobbies=["篮球", "街舞", "潮流球鞋", "嘻哈音乐"],
        landmines=["讨厌说话不回/冷暴力", "排斥过度打探隐私"],
        weaknesses=["胜负欲太强/容易较真", "作息极其不规律/熬夜党"],
        personality=["热情豪爽", "行动派"],
        cv_scene="篮球场",
        cv_objects=["篮球", "运动护腕"]
    ),
    UserProfile(
        user_id="user_005",
        nickname="许飞扬",
        hobbies=["飞盘", "露营", "公路骑行", "Vlog剪辑"],
        landmines=["讨厌负能量爆棚"],
        weaknesses=["自来熟容易吓到社恐", "容易三分钟热度"],
        personality=["破冰高手", "极具团队感染力"],
        cv_scene="校园草坪/操场",
        cv_objects=["飞盘", "骑行头盔"]
    ),
    UserProfile(
        user_id="user_006",
        nickname="宋晴空",
        hobbies=["排球", "游泳", "户外徒步", "播客"],
        landmines=["极其讨厌别人迟到"],
        weaknesses=["对细节过于挑剔", "不擅长拒绝别人"],
        personality=["沉稳靠谱", "非常有韧性"],
        cv_scene="体育馆/排球场",
        cv_objects=["排球", "运动饮料"]
    ),
    UserProfile(
        user_id="user_007",
        nickname="高远航",
        hobbies=["轮滑", "滑板", "摇滚乐", "夜跑"],
        landmines=["讨厌社交大吵大闹"],
        weaknesses=["熟人疯子/生人高冷", "作息极其不规律/熬夜党"],
        personality=["酷飒洒脱", "学习能力强"],
        cv_scene="校园广场",
        cv_objects=["滑板", "蓝牙音箱"]
    ),
    UserProfile(
        user_id="user_008",
        nickname="袁劲松",
        hobbies=["乒乓球", "棋牌", "跑步", "历史故事"],
        landmines=["讨厌说话不回/冷暴力", "讨厌负能量爆棚"],
        weaknesses=["选择困难症", "说话过于直白/直男发言"],
        personality=["踏实勤恳", "情绪稳定"],
        cv_scene="体育馆/乒乓球室",
        cv_objects=["乒乓球拍", "毛巾"]
    ),
    UserProfile(
        user_id="user_009",
        nickname="蒋晨霞",
        hobbies=["瑜伽", "普拉提", "健康轻食", "古典乐"],
        landmines=["讨厌社交大吵大闹", "排斥过度打探隐私"],
        weaknesses=["重度社恐/极度怕生", "容易焦虑/情绪化"],
        personality=["温柔专注", "生活自律"],
        cv_scene="舞蹈房/瑜伽室",
        cv_objects=["瑜伽垫", "保温杯"]
    ),

    # ---------------- 2. 图书馆与学习硬核类 (user_010 ~ user_018) ----------------
    UserProfile(
        user_id="user_010",
        nickname="张晨阳",
        hobbies=["羽毛球", "开源项目", "摇滚乐", "食堂火锅"],
        landmines=["极其讨厌别人迟到", "讨厌说话不回/冷暴力"],
        weaknesses=["拖延症/经常赶截止时间", "选择困难症"],
        personality=["社交气氛担当", "幽默大方"],
        cv_scene="图书馆/自习室",
        cv_objects=["赶作业", "互相监督"]
    ),
    UserProfile(
        user_id="user_011",
        nickname="周雅婷",
        hobbies=["看书/阅读", "周杰伦", "烘焙", "自习搭子"],
        landmines=["讨厌负能量爆棚"],
        weaknesses=["选择困难症", "社恐/不善言辞"],
        personality=["性格极温和", "非常有耐心"],
        cv_scene="图书馆/书店",
        cv_objects=["高数习题册", "笔记本电脑"]
    ),
    UserProfile(
        user_id="user_012",
        nickname="许博文",
        hobbies=["考研刷题", "哲学思考", "古典乐", "咖啡"],
        landmines=["讨厌社交大吵大闹", "排斥过度打探隐私"],
        weaknesses=["表达过于严肃/不够幽默", "容易焦虑/情绪化"],
        personality=["学识渊博", "思想深邃"],
        cv_scene="图书馆/自习室",
        cv_objects=["学术著作", "草稿纸"]
    ),
    UserProfile(
        user_id="user_013",
        nickname="韩清静",
        hobbies=["日语学习", "动漫插画", "静音自习"],
        landmines=["讨厌社交大吵大闹", "极其讨厌别人迟到"],
        weaknesses=["重度社恐/极度怕生", "不擅长拒绝别人"],
        personality=["极其安静不打扰人", "贴心细腻"],
        cv_scene="图书馆/自习室",
        cv_objects=["iPad画板", "消音耳机"]
    ),
    UserProfile(
        user_id="user_014",
        nickname="方文思",
        hobbies=["保研准备", "Python数据分析", "英文原著", "黑咖啡"],
        landmines=["极其讨厌别人迟到", "讨厌负能量爆棚"],
        weaknesses=["过于理性/不够圆滑", "拖延症/经常赶截止时间"],
        personality=["逻辑严密", "高效专注"],
        cv_scene="图书馆/电子阅览室",
        cv_objects=["平板电脑", "Kindle"]
    ),
    UserProfile(
        user_id="user_015",
        nickname="任雪纯",
        hobbies=["法学案例", "辩论", "纪录片", "手帐"],
        landmines=["讨厌说话不回/冷暴力", "讨厌社交大吵大闹"],
        weaknesses=["胜负欲太强/容易较真", "对细节过于挑剔"],
        personality=["正义感强", "口齿伶俐"],
        cv_scene="图书馆/研讨室",
        cv_objects=["法律汇编", "荧光笔"]
    ),
    UserProfile(
        user_id="user_016",
        nickname="魏墨林",
        hobbies=["书法", "古籍研究", "围棋", "饮茶"],
        landmines=["讨厌社交大吵大闹", "排斥过度打探隐私"],
        weaknesses=["表达过于严肃/不够幽默", "社恐/不善言辞"],
        personality=["儒雅沉稳", "心境平和"],
        cv_scene="图书馆/古籍阅览室",
        cv_objects=["毛笔", "茶杯"]
    ),
    UserProfile(
        user_id="user_017",
        nickname="廖子健",
        hobbies=["建模比赛", "LaTeX排版", "高等代数", "能量饮料"],
        landmines=["极其讨厌别人迟到"],
        weaknesses=["作息极其不规律/熬夜党", "说话过于直白/直男发言"],
        personality=["解题能手", "极具探索欲"],
        cv_scene="自习室/实验楼",
        cv_objects=["计算器", "草稿纸"]
    ),
    UserProfile(
        user_id="user_018",
        nickname="苏微澜",
        hobbies=["心理学", "自我提升", "植物学", "轻音乐"],
        landmines=["讨厌负能量爆棚", "排斥过度打探隐私"],
        weaknesses=["选择困难症", "容易焦虑/情绪化"],
        personality=["善解人意", "共情力强"],
        cv_scene="图书馆/露天中庭",
        cv_objects=["心理学著作", "多肉植物"]
    ),

    # ---------------- 3. 美食与生活搭子类 (user_019 ~ user_027) ----------------
    UserProfile(
        user_id="user_019",
        nickname="陈思涵",
        hobbies=["猫咪/宠物", "摄影", "校园咖啡厅", "独立音乐"],
        landmines=["排斥过度打探隐私", "讨厌社交大吵大闹"],
        weaknesses=["容易焦虑/情绪化", "重度颜控/外貌协会"],
        personality=["审美极佳", "心思细腻"],
        cv_scene="校园咖啡厅",
        cv_objects=["撸猫", "喝咖啡"]
    ),
    UserProfile(
        user_id="user_020",
        nickname="刘胖肉",
        hobbies=["食堂火锅", "自助餐", "美食探店", "桌游"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["选择困难症", "拖延症/经常赶截止时间"],
        personality=["幽默随和", "绝不扫兴"],
        cv_scene="食堂火锅/餐饮",
        cv_objects=["火锅餐券", "饮料"]
    ),
    UserProfile(
        user_id="user_021",
        nickname="赵甜甜",
        hobbies=["下午茶", "汉服拍照", "流行音乐", "烘焙"],
        landmines=["讨厌负能量爆棚", "排斥过度打探隐私"],
        weaknesses=["熟人疯子/生人高冷", "容易三分钟热度"],
        personality=["拍照技术好", "分享欲强"],
        cv_scene="校园咖啡厅",
        cv_objects=["蛋糕", "相机"]
    ),
    UserProfile(
        user_id="user_022",
        nickname="龚小厨",
        hobbies=["宿舍烹饪", "菜市场探店", "美食Vlog", "烧烤"],
        landmines=["讨厌说话不回/冷暴力", "极其讨厌别人迟到"],
        weaknesses=["对细节过于挑剔", "拖延症/经常赶截止时间"],
        personality=["厨艺高超", "热情大方"],
        cv_scene="学生食堂/美食街",
        cv_objects=["小吃拼盘", "餐巾纸"]
    ),
    UserProfile(
        user_id="user_023",
        nickname="唐悠悠",
        hobbies=["奶茶鉴赏", "逛街购物", "韩剧", "探店打卡"],
        landmines=["讨厌负能量爆棚"],
        weaknesses=["重度颜控/外貌协会", "选择困难症"],
        personality=["活力四射", "性格开朗"],
        cv_scene="校园商业街",
        cv_objects=["奶茶", "购物袋"]
    ),
    UserProfile(
        user_id="user_024",
        nickname="彭可口",
        hobbies=["螺蛳粉", "夜市摊", "整理打扫", "游戏主播"],
        landmines=["讨厌社交大吵大闹"],
        weaknesses=["熟人疯子/生人高冷", "作息极其不规律/熬夜党"],
        personality=["真实不装", "特别好相处"],
        cv_scene="宿舍/夜市",
        cv_objects=["螺蛳粉包", "可乐"]
    ),
    UserProfile(
        user_id="user_025",
        nickname="崔甜圈",
        hobbies=["手工冰淇淋", "甜品制作", "插画", "迪士尼"],
        landmines=["排斥过度打探隐私", "讨厌负能量爆棚"],
        weaknesses=["社恐/不善言辞", "不擅长拒绝别人"],
        personality=["甜美治愈", "很有爱心"],
        cv_scene="甜品店/烘焙坊",
        cv_objects=["马卡龙", "围裙"]
    ),
    UserProfile(
        user_id="user_026",
        nickname="贺微醺",
        hobbies=["精酿啤酒", "露天小酒馆", "爵士乐", "聊人生"],
        landmines=["讨厌说话不回/冷暴力", "讨厌社交大吵大闹"],
        weaknesses=["容易焦虑/情绪化", "表达过于严肃/不够幽默"],
        personality=["故事丰富", "善于深度沟通"],
        cv_scene="校园小酒馆/露台",
        cv_objects=["精酿啤酒杯", "花生米"]
    ),
    UserProfile(
        user_id="user_027",
        nickname="石饱饱",
        hobbies=["早茶", "广式点心", "纪录片", "养生"],
        landmines=["极其讨厌别人迟到"],
        weaknesses=["选择困难症", "过于理性/不够圆滑"],
        personality=["稳重包容", "会照顾人"],
        cv_scene="学生食堂",
        cv_objects=["蒸笼", "保温杯"]
    ),

    # ---------------- 4. 极客与艺术创作类 (user_028 ~ user_036) ----------------
    UserProfile(
        user_id="user_028",
        nickname="宋无界",
        hobbies=["AI 绘画", "GameJam开发", "硬核科幻", "赛博朋克"],
        landmines=["讨厌社交大吵大闹"],
        weaknesses=["作息极其不规律/熬夜党", "说话过于直白/直男发言"],
        personality=["技术大牛", "专注度极高"],
        cv_scene="教室/自习室",
        cv_objects=["机械键盘", "多屏显示器"]
    ),
    UserProfile(
        user_id="user_029",
        nickname="姜雨晴",
        hobbies=["吉他弹唱", "独立乐队", "胶片摄影", "黑胶唱片"],
        landmines=["排斥过度打探隐私", "讨厌负能量爆棚"],
        weaknesses=["熟人疯子/生人高冷", "情绪起伏大"],
        personality=["很有艺术气质", "极具个性"],
        cv_scene="校园草坪/公园",
        cv_objects=["木吉他", "胶片机"]
    ),
    UserProfile(
        user_id="user_030",
        nickname="钟极客",
        hobbies=["Linux内核", "网络安全", "树莓派", "开源社区"],
        landmines=["讨厌说话不回/冷暴力", "排斥过度打探隐私"],
        weaknesses=["重度社恐/极度怕生", "说话过于直白/直男发言"],
        personality=["硬核技术控", "严谨专注"],
        cv_scene="创客空间/实验室",
        cv_objects=["代码界面", "电路板"]
    ),
    UserProfile(
        user_id="user_031",
        nickname="莫剪辑",
        hobbies=["B站UP主", "特效合成", "电影达人", "赛车"],
        landmines=["极其讨厌别人迟到"],
        weaknesses=["拖延症/经常赶截止时间", "作息极其不规律/熬夜党"],
        personality=["脑洞极大", "审美在线"],
        cv_scene="多媒体教室/宿舍",
        cv_objects=["剪辑软件", "手绘板"]
    ),
    UserProfile(
        user_id="user_032",
        nickname="戴美学",
        hobbies=["3D建模", "UI设计", "现代艺术展", "建筑设计"],
        landmines=["讨厌社交大吵大闹", "讨厌负能量爆棚"],
        weaknesses=["重度颜控/外貌协会", "对细节过于挑剔"],
        personality=["完美主义", "有独特品味"],
        cv_scene="艺术设计楼/展厅",
        cv_objects=["手绘笔", "设计草图"]
    ),
    UserProfile(
        user_id="user_033",
        nickname="薛音符",
        hobbies=["钢琴演奏", "电子音乐编曲", "声乐", "音乐剧"],
        landmines=["排斥过度打探隐私"],
        weaknesses=["情绪起伏大", "容易焦虑/情绪化"],
        personality=["富有浪漫情怀", "灵感丰富"],
        cv_scene="琴房/音乐厅",
        cv_objects=["钢琴", "五线谱"]
    ),
    UserProfile(
        user_id="user_034",
        nickname="阎次元",
        hobbies=["Cosplay", "二次元手办", "宅舞", "声优"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["熟人疯子/生人高冷", "自来熟容易吓到社恐"],
        personality=["表现力极强", "真诚热情"],
        cv_scene="活动中心/社团舞台",
        cv_objects=["Cos服饰", "化妆镜"]
    ),
    UserProfile(
        user_id="user_035",
        nickname="傅镜头",
        hobbies=["人像摄影", "无人机航拍", "调色", "旅行"],
        landmines=["极其讨厌别人迟到", "讨厌负能量爆棚"],
        weaknesses=["胜负欲太强/容易较真", "拖延症/经常赶截止时间"],
        personality=["观察力敏锐", "有执行力"],
        cv_scene="校园天台/景致点",
        cv_objects=["单反相机", "无人机遥控器"]
    ),
    UserProfile(
        user_id="user_036",
        nickname="尹创客",
        hobbies=["3D打印", "机器人竞赛", "智能家居", "乐高"],
        landmines=["讨厌社交大吵大闹"],
        weaknesses=["过于理性/不够圆滑", "表达过于严肃/不够幽默"],
        personality=["动手能力极强", "逻辑清晰"],
        cv_scene="工程训练中心",
        cv_objects=["3D打印机", "螺丝刀"]
    ),

    # ---------------- 5. 休闲娱乐与社团搭子类 (user_037 ~ user_045) ----------------
    UserProfile(
        user_id="user_037",
        nickname="郑乐天",
        hobbies=["剧本杀", "阿瓦隆桌游", "电影鉴赏", "脱口秀"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["自来熟容易吓到社恐", "三分钟热度"],
        personality=["破冰大师", "接话能手"],
        cv_scene="校园公共区域",
        cv_objects=["桌游卡牌", "零食"]
    ),
    UserProfile(
        user_id="user_038",
        nickname="沈小白",
        hobbies=["散步", "逛校园看猫", "轻音乐", "慢节奏生活"],
        landmines=["讨厌社交大吵大闹", "极其讨厌别人迟到"],
        weaknesses=["社恐/不善言辞", "选择困难症"],
        personality=["治愈系性格", "倾听者"],
        cv_scene="校园草坪/公园",
        cv_objects=["猫粮", "长椅"]
    ),
    UserProfile(
        user_id="user_039",
        nickname="魏星洲",
        hobbies=["观星/天文", "夜间散步", "科幻小说", "摄影"],
        landmines=["排斥过度打探隐私"],
        weaknesses=["表达过于严肃/不够幽默", "作息极其不规律/熬夜党"],
        personality=["浪漫主义者", "耐心十足"],
        cv_scene="操场看台/夜间",
        cv_objects=["望远镜", "三脚架"]
    ),
    UserProfile(
        user_id="user_040",
        nickname="卢影迷",
        hobbies=["艺术电影", "电影院打卡", "剧本创作", "影评"],
        landmines=["讨厌社交大吵大闹", "排斥过度打探隐私"],
        weaknesses=["重度社恐/极度怕生", "容易焦虑/情绪化"],
        personality=["感受力细腻", "思考深入"],
        cv_scene="学生活动中心/放映厅",
        cv_objects=["电影票存根", "爆米花"]
    ),
    UserProfile(
        user_id="user_041",
        nickname="贾魔术",
        hobbies=["近景魔术", "花式扑克", "解谜游戏", "心理学"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["熟人疯子/生人高冷", "胜负欲太强/容易较真"],
        personality=["幽默风趣", "充满神秘感"],
        cv_scene="社团活动室",
        cv_objects=["扑克牌", "魔术道具"]
    ),
    UserProfile(
        user_id="user_042",
        nickname="邹竞技",
        hobbies=["英雄联盟", "王者荣耀", "电竞比赛", "开黑"],
        landmines=["讨厌说话不回/冷暴力", "极其讨厌别人迟到"],
        weaknesses=["胜负欲太强/容易较真", "作息极其不规律/熬夜党"],
        personality=["抗压能力强", "团队协作好"],
        cv_scene="电竞社/宿舍",
        cv_objects=["电竞耳机", "游戏键盘"]
    ),
    UserProfile(
        user_id="user_043",
        nickname="秦演讲",
        hobbies=["主持", "辩论队", "演讲", "朗诵"],
        landmines=["讨厌负能量爆棚"],
        weaknesses=["自来熟容易吓到社恐", "对细节过于挑剔"],
        personality=["台风稳健", "感染力极强"],
        cv_scene="报告厅/讲台",
        cv_objects=["麦克风", "演讲稿"]
    ),
    UserProfile(
        user_id="user_044",
        nickname="邱有爱",
        hobbies=["流浪动物救助", "志愿服务", "环保", "公益"],
        landmines=["讨厌负能量爆棚", "讨厌说话不回/冷暴力"],
        weaknesses=["不擅长拒绝别人", "容易焦虑/情绪化"],
        personality=["极具爱心", "正能量满满"],
        cv_scene="校园小树林/动物救助站",
        cv_objects=["猫罐头", "志愿者红马甲"]
    ),
    UserProfile(
        user_id="user_045",
        nickname="易玩偶",
        hobbies=["盲盒抽卡", "抓娃娃", "乐高拼搭", "粘土制作"],
        landmines=["排斥过度打探隐私"],
        weaknesses=["选择困难症", "容易三分钟热度"],
        personality=["童心未泯", "乐观开朗"],
        cv_scene="学生街/展示柜",
        cv_objects=["盲盒公仔", "手工艺品"]
    ),

    # ---------------- 6. 综合跨界与斜杠青年类 (user_046 ~ user_050) ----------------
    UserProfile(
        user_id="user_046",
        nickname="骆斜杠",
        hobbies=["编程", "吉他", "马拉松", "创业"],
        landmines=["极其讨厌别人迟到", "讨厌说话不回/冷暴力"],
        weaknesses=["作息极其不规律/熬夜党", "过于理性/不够圆滑"],
        personality=["精力充沛", "目标明确"],
        cv_scene="创业孵化基地",
        cv_objects=["白板", "商业计划书"]
    ),
    UserProfile(
        user_id="user_047",
        nickname="陶国风",
        hobbies=["汉服秀", "民乐古筝", "茶道", "国漫"],
        landmines=["讨厌社交大吵大闹", "排斥过度打探隐私"],
        weaknesses=["重度颜控/外貌协会", "社恐/不善言辞"],
        personality=["温婉典雅", "做事专注"],
        cv_scene="校园传统文化社/庭院",
        cv_objects=["古筝", "汉服发饰"]
    ),
    UserProfile(
        user_id="user_048",
        nickname="温旅行",
        hobbies=["穷游背包客", "地理风土", "摄影", "语言学习"],
        landmines=["讨厌负能量爆棚"],
        weaknesses=["拖延症/经常赶截止时间", "容易三分钟热度"],
        personality=["见多识广", "适应能力极强"],
        cv_scene="校园主干道/车站",
        cv_objects=["登山包", "地图"]
    ),
    UserProfile(
        user_id="user_049",
        nickname="夏静禅",
        hobbies=["冥想", "徒步", "写作", "独处"],
        landmines=["讨厌社交大吵大闹", "排斥过度打探隐私"],
        weaknesses=["重度社恐/极度怕生", "表达过于严肃/不够幽默"],
        personality=["超然沉静", "深度思考"],
        cv_scene="校园人工湖畔",
        cv_objects=["笔记本", "降噪耳机"]
    ),
    UserProfile(
        user_id="user_050",
        nickname="祁高能",
        hobbies=["街舞Breaking", "滑雪", "DJ打碟", "健身"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["胜负欲太强/容易较真", "熟人疯子/生人高冷"],
        personality=["舞台王者", "极具爆发力"],
        cv_scene="排练厅/镜面室",
        cv_objects=["便携音箱", "鸭舌帽"]
    )
]


def get_all_mock_users() -> list[UserProfile]:
    """获取所有模拟用户列表。"""
    return MOCK_USERS


def get_user_by_id(user_id: str) -> UserProfile | None:
    """根据 user_id 查找指定用户。"""
    for u in MOCK_USERS:
        if u.user_id == user_id:
            return u
    return None