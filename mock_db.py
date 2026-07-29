# mock_db.py
from user_profile import UserProfile

MOCK_USERS = [
    # ---------------- 1. 运动与户外类 ----------------
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

    # ---------------- 2. 图书馆与学习硬核类 ----------------
    UserProfile(
        user_id="user_004",
        nickname="张晨阳",
        hobbies=["羽毛球", "开源项目", "摇滚乐", "食堂火锅"],
        landmines=["极其讨厌别人迟到", "讨厌说话不回/冷暴力"],
        weaknesses=["拖延症/经常赶截止时间", "选择困难症"],
        personality=["社交气氛担当", "幽默大方"],
        cv_scene="图书馆/自习室",
        cv_objects=["赶作业", "互相监督"]
    ),
    UserProfile(
        user_id="user_005",
        nickname="周雅婷",
        hobbies=["看书/阅读", "周杰伦", "烘焙", "自习搭子"],
        landmines=["讨厌负能量爆棚"],
        weaknesses=["选择困难症", "社恐/不善言辞"],
        personality=["性格极温和", "非常有耐心"],
        cv_scene="图书馆/书店",
        cv_objects=["高数习题册", "笔记本电脑"]
    ),
    UserProfile(
        user_id="user_006",
        nickname="许博文",
        hobbies=["考研刷题", "哲学思考", "古典乐", "咖啡"],
        landmines=["讨厌社交大吵大闹", "排斥过度打探隐私"],
        weaknesses=["表达过于严肃/不够幽默", "容易焦虑/情绪化"],
        personality=["学识渊博", "思想深邃"],
        cv_scene="图书馆/自习室",
        cv_objects=["学术著作", "草稿纸"]
    ),
    UserProfile(
        user_id="user_007",
        nickname="韩清静",
        hobbies=["日语学习", "动漫插画", "静音自习"],
        landmines=["讨厌社交大吵大闹", "极其讨厌别人迟到"],
        weaknesses=["重度社恐/极度怕生", "不擅长拒绝别人"],
        personality=["极其安静不打扰人", "贴心细腻"],
        cv_scene="图书馆/自习室",
        cv_objects=["iPad画板", "消音耳机"]
    ),

    # ---------------- 3. 美食与生活搭子类 ----------------
    UserProfile(
        user_id="user_008",
        nickname="陈思涵",
        hobbies=["猫咪/宠物", "摄影", "校园咖啡厅", "独立音乐"],
        landmines=["排斥过度打探隐私", "讨厌社交大吵大闹"],
        weaknesses=["容易焦虑/情绪化", "重度颜控/外貌协会"],
        personality=["审美极佳", "心思细腻"],
        cv_scene="校园咖啡厅",
        cv_objects=["撸猫", "喝咖啡"]
    ),
    UserProfile(
        user_id="user_009",
        nickname="刘胖肉",
        hobbies=["食堂火锅", "自助餐", "美食探店", "桌游"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["选择困难症", "拖延症/经常赶截止时间"],
        personality=["幽默随和", "绝不扫兴"],
        cv_scene="食堂火锅/餐饮",
        cv_objects=["火锅餐券", "饮料"]
    ),
    UserProfile(
        user_id="user_010",
        nickname="赵甜甜",
        hobbies=["下午茶", "汉服拍照", "流行音乐", "烘焙"],
        landmines=["讨厌负能量爆棚", "排斥过度打探隐私"],
        weaknesses=["熟人疯子/生人高冷", "容易三分钟热度"],
        personality=["拍照技术好", "分享欲强"],
        cv_scene="校园咖啡厅",
        cv_objects=["蛋糕", "相机"]
    ),

    # ---------------- 4. 极客与艺术创作类 ----------------
    UserProfile(
        user_id="user_011",
        nickname="宋无界",
        hobbies=["AI 绘画", "GameJam开发", "硬核科幻", "赛博朋克"],
        landmines=["讨厌社交大吵大闹"],
        weaknesses=["作息极其不规律/熬夜党", "说话过于直白/直男发言"],
        personality=["技术大牛", "专注度极高"],
        cv_scene="教室/自习室",
        cv_objects=["机械键盘", "多屏显示器"]
    ),
    UserProfile(
        user_id="user_012",
        nickname="姜雨晴",
        hobbies=["吉他弹唱", "独立乐队", "胶片摄影", "黑胶唱片"],
        landmines=["排斥过度打探隐私", "讨厌负能量爆棚"],
        weaknesses=["熟人疯子/生人高冷", "情绪起伏大"],
        personality=["很有艺术气质", "极具个性"],
        cv_scene="校园草坪/公园",
        cv_objects=["木吉他", "胶片机"]
    ),

    # ---------------- 5. 休闲娱乐与社团搭子类 ----------------
    UserProfile(
        user_id="user_013",
        nickname="郑乐天",
        hobbies=["剧本杀", "阿瓦隆桌游", "电影鉴赏", "脱口秀"],
        landmines=["讨厌说话不回/冷暴力"],
        weaknesses=["自来熟容易吓到社恐", "三分钟热度"],
        personality=["破冰大师", "接话能手"],
        cv_scene="校园公共区域",
        cv_objects=["桌游卡牌", "零食"]
    ),
    UserProfile(
        user_id="user_014",
        nickname="沈小白",
        hobbies=["散步", "逛校园看猫", "轻音乐", "慢节奏生活"],
        landmines=["讨厌社交大吵大闹", "极其讨厌别人迟到"],
        weaknesses=["社恐/不善言辞", "选择困难症"],
        personality=["治愈系性格", "倾听者"],
        cv_scene="校园草坪/公园",
        cv_objects=["猫粮", "长椅"]
    ),
    UserProfile(
        user_id="user_015",
        nickname="魏星洲",
        hobbies=["观星/天文", "夜间散步", "科幻小说", "摄影"],
        landmines=["排斥过度打探隐私"],
        weaknesses=["表达过于严肃/不够幽默", "作息极其不规律/熬夜党"],
        personality=["浪漫主义者", "耐心十足"],
        cv_scene="操场看台/夜间",
        cv_objects=["望远镜", "三脚架"]
    )
]

def get_all_mock_users() -> list[UserProfile]:
    return MOCK_USERS

def get_user_by_id(user_id: str) -> UserProfile | None:
    for u in MOCK_USERS:
        if u.user_id == user_id:
            return u
    return None