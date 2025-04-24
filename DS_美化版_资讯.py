# ================== 导入依赖 ==================
import streamlit as st
import pandas as pd
import numpy as np
import os
import secrets
import string
import datetime
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from dateutil.parser import parse
import requests
from transformers import pipeline

# ================== 全局常量 ==================
# 自定义CSS样式
STYLE = """
<style>
/* 基础主题色 */
:root {
    --primary: #2c3e50;
    --secondary: #3498db;
    --accent: #e74c3c;
    --background: #f8f9fa;
    --text: #2c3e50;
}

/* 整体布局优化 */
.main {
    background-color: var(--background);
}

/* 标题样式 */
.header {
    font-size: 2.2rem;
    color: var(--primary);
    text-align: center;
    margin: 1.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid var(--secondary);
}

/* 分类卡片优化 */
.item-card {
    padding: 1.2rem;
    margin: 1rem 0;
    border-radius: 12px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: none;
}

.item-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

/* 链接按钮样式 */
.link-btn {
    color: var(--secondary) !important;
    font-weight: 500;
    transition: color 0.2s;
}

.link-btn:hover {
    color: var(--primary) !important;
}

/* 徽章样式统一 */
.tutorial-badge {
    font-size: 0.75em;
    padding: 4px 12px;
    border-radius: 20px;
    margin-left: 8px;
}

.popularity-badge {
    background: linear-gradient(135deg, #ff6b6b, #ff8e53);
    color: white;
    font-size: 0.85em;
    padding: 4px 12px;
    border-radius: 20px;
}

/* 侧边栏优化 */
[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #f8f9fa 0%, #e9ecef 100%);
    border-right: 1px solid #dee2e6;
}

/* 输入框样式 */
.stTextInput input {
    border-radius: 8px;
    border: 1px solid #dee2e6;
}

/* 按钮样式 */
.stButton>button {
    border-radius: 8px;
    background: var(--secondary);
    color: white;
    transition: all 0.2s;
}

.stButton>button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* 统计卡片 */
.stMetric {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

/* 响应式优化 */
@media (max-width: 768px) {
    .item-card {
        padding: 1rem;
        margin: 0.8rem 0;
    }

    .header {
        font-size: 1.8rem;
    }

    [data-testid="stSidebar"] {
        width: 240px !important;
    }
}

/* 工具提示优化 */
.tooltip .tooltiptext {
    width: 320px;
    background: rgba(255,255,240,0.98);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    border: 1px solid rgba(0,0,0,0.08);
    backdrop-filter: blur(4px);
}

/* 渐变标题 */
.gradient-text {
    background: linear-gradient(45deg, #2c3e50, #3498db);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: 700;
}
/* 新增工具提示样式 */
.item-card {
    position: relative;
    cursor: pointer;
}

.item-card .tooltip {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    z-index: 999;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255,255,240,0.98);
    color: #2c3e50;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    border: 1px solid rgba(0,0,0,0.08);
    min-width: 280px;
    max-width: 400px;
    transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55);
}

.item-card:hover .tooltip {
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(-5px);
}

.tooltip-content {
    font-size: 0.9em;
    line-height: 1.6;
}

.tooltip h4 {
    margin: 0 0 8px 0;
    color: var(--secondary);
    font-size: 1.1em;
}

.tooltip .description {
    margin-bottom: 8px;
}

.tooltip .features {
    color: #666;
    font-size: 0.9em;
}
/* 新增热点资讯样式 */
.news-ticker {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    position: relative;
    overflow: hidden;
}

.news-marquee {
    display: flex;
    animation: scroll 25s linear infinite;
    gap: 2rem;
}

@keyframes scroll {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

.news-card {
    flex: 0 0 300px;
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    transition: transform 0.3s ease;
    border-left: 4px solid var(--secondary);
}

.news-card:hover {
    transform: translateY(-3px);
}

.news-title {
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 0.5rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.news-source {
    font-size: 0.8em;
    color: #666;
    margin-bottom: 0.5rem;
}

.news-snippet {
    font-size: 0.9em;
    color: #666;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.news-time {
    font-size: 0.8em;
    color: #999;
    margin-top: 0.5rem;
}
</style>
"""


# 国家与国旗映射
COUNTRY_FLAGS = {
    '美国': '🇺🇸', '中国': '🇨🇳', '法国': '🇫🇷', '以色列': '🇮🇱',
    '英国': '🇬🇧', '加拿大': '🇨🇦', '日本': '🇯🇵', '德国': '🇩🇪',
    '印度': '🇮🇳', '俄罗斯': '🇷🇺', '韩国': '🇰🇷', '澳大利亚': '🇦🇺', '意大利': '🇮🇹',
 '新加坡': '🇸🇬',
}

CATEGORY_STRUCTURE = {
    "✍️AI写作工具":['学术论文写作', '公文/政务写作', '小说/文学创作', '营销/商业文案', '多语言写作辅助', '综合内容创作', '长文生成工具', '智能改写优化', '垂直领域工具', '国际知名平台'],
    "🖼️AI图像生成与创作类工具":["基础文生图","垂直场景生成","视频生成"],
    "🖼️AI图像编辑处理类工具":["专业图像处理","智能修复","多功能处理套件","AI图片背景移除工具","AI图片物体抹除","AI图像编辑"],
    "🖼️AI图像特殊、特色工能工具":["人像处理","商业应用","跨模态","图像翻译","实时生成","企业级解决方案","AI图片无损放大"],
    "🖼️AI图像社区与模型平台":["模型分享社区","在线创作社区"],
    "🖼️AI图像移动端优化":["轻量级应用","移动端其他"],
    "🖼️AI图片优化修复":["老照片修复专项","批量处理工具","智能修图软件","上色与调色工具","图片增强与清晰化","集成生成与修图","专业领域工具"],
    "🎥AI视频工具":["文生视频","短视频生成","视频编辑与增强","数字人视频生成","多模态生成","商业视频创作","开源/大模型平台","动画/动漫视频","影视级制作","国际知名平台（视频）","特色功能"],
    "📡AI办公工具":["PPT智能生成","图表与数据可视化","文档智能处理","国际知名平台","教学专用","特色功能","效率增强","AI会议工具"],
    "🎨AI设计工具":["电商设计","平面设计","UX/UI设计","3D/模型设计","综合设计平台","AI图像编辑","专业领域","国际知名平台（设计）"],
    "💬AI对话聊天":["通用型智能助手","角色扮演/虚拟陪伴","企业大模型产品","国际知名平台（对话聊天）","多模态助手",],
    "💻AI编程工具":["代码生成与补全","代码测试与分析","云端与在线开发","前端与UI开发", "开源与自托管","企业级解决方案（编程）","IDE集成","智能开发辅助","垂直领域","国际知名平台（编程）"],
    "🔍AI搜索引擎":["通用综合搜索","垂直领域搜索【学术研究】","垂直领域搜索【商业金融】","垂直领域搜索【开发者专用】","交互模式创新","多媒体搜索","生态集成搜索","场景化搜索","技术架构创新","国际平台"],
    '🎵AI音频工具':["文本转语音（TTS）","语音转文字（SST）","音乐生成平台","声音克隆与变声","音频处理与编辑","歌声合成系统","语音交互","开源与开发者","多模态创新平台","企业级解决方案（音频）"],
    '🛠️AI开发平台':["智能体开发平台","深度学习框架","机器学习库与工具","无代码/低代码平台","企业级解决方案（开发平台）","数据标注与处理","自然语言处理（NLP）","边缘计算与部署","开发者社区与协作","综合服务平台"],
    '⚙️AI训练模型':["大语言模型（LLM）","多模态大模型","图像生成模型","视频生成模型","代码生成模型","开源框架与工具","企业级平台","本地部署","训练优化","模型库与社区","专项领域模型","学术研究项目"],
    '📑AI内容检测':["通用文本检测","学术专用检测","多语种检测","图像检测","原创性分析","GPT专项检测","企业级解决方案（检测）"],
    '🌍AI语言翻译':["通用文本翻译","多语言支持","企业级解决方案（翻译）","实时语音翻译","多模态翻译","浏览器插件","同声传译","特色功能"],
    '🏛️AI法律助手':["法律知识检索平台","法律咨询与建议助手", "合同生成与管理", "开源法律模型与框架"],
    '📡AI提示指令':["提示词生成与优化","垂直领域提示库","社区与市场平台", "开源框架与开发者", "教育学习资源", "浏览器集成","多模态支持"],
    '📦AI模型评测':["综合能力评测基准","中文专项测评体系","多模态评测平台","垂直领域评测基准","开源社区排行榜","学术机构评测体系","评测方法论创新"],
    '👁️AI学习网站':["学术机构课程","企业培训体系","MOOC平台专区","开源学习社区","实践实训平台","青少年教育专区","通识教育平台","工具型学习辅助","阅读提升","垂直领域【教育】"],
}

COUNTRY_MAPPING = {
                '中国': ['china', 'cn', '中国', '中華', '中国大陆'],
                '美国': ['usa', 'us', 'america', '美国', '美利坚'],
                '日本': ['japan', 'jp', '日本', '日本国'],
                '韩国': ['korea', 'kr', '韩国', '大韩民国', 'korean'],
                '英国': ['uk', 'gb', 'united kingdom', '英国', '大不列颠', '英伦'],
                '法国': ['france', 'fr', '法国', '法兰西'],
                '德国': ['germany', 'de', '德国', '德意志'],
                '意大利': ['italy', 'it', '意大利', '意大利共和国'],
                '加拿大': ['canada', 'ca', '加拿大'],
                '澳大利亚': ['australia', 'au', '澳大利亚'],
                '新西兰': ['new zealand', 'nz', '新西兰'],
                '瑞士': ['switzerland', 'ch', '瑞士', '瑞士联邦'],
                '荷兰': ['netherlands', 'nl', '荷兰', '尼德兰'],
                '比利时': ['belgium', 'be', '比利时'],
                '奥地利': ['austria', 'at', '奥地利'],
                '瑞典': ['sweden', 'se', '瑞典'],
                '挪威': ['norway', 'no', '挪威'],
                '丹麦': ['denmark', 'dk', '丹麦'],
                '芬兰': ['finland', 'fi', '芬兰'],
                '爱尔兰': ['ireland', 'ie', '爱尔兰'],
                '卢森堡': ['luxembourg', '.lu', '卢森堡'],
                '西班牙': ['spain', '.es', '西班牙'],
                '葡萄牙': ['portugal', '.pt', '葡萄牙'],
                '希腊': ['greece', '.gr', '希腊'],
                '以色列': ['israel', '.il', '以色列'],
                '新加坡': ['singapore', '.sg', '新加坡'],
                '波兰': ['poland', '.pl', '波兰'],
            }

# ================== 在CATEGORY_STRUCTURE之后添加 ==================
BOOKMARK_CATEGORIES = {
    "常用工具": ["AI办公","AI视频","AI图像","AI编程","工具大全"],
    "开发资源": ["前端", "后端", "数据库", "云服务","大模型调用"],
    "学习平台": ["公开课", "技术文档", "行业报告"],
    "设计资源": ["UI模板", "图标库", "配色方案"],
    "研究机构": ["AI实验室", "大学研究", "创新中心"]
}

# 在全局常量部分新增
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
NEWS_API_KEY = "c15bca5299c84e97be7f3c7fe3678fe8"  # 实际使用时需替换为有效API密钥


# ================== 配置类 ==================
class EnhancedConfig:
    MIN_POPULARITY = 0
    SERPER_API_KEY = 'e89573bc1ad5b0c3fa6b58bc1dd3fcc8b585bf6a'

class PathConfig:
    """统一管理所有文件路径"""
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = Path(os.getenv("DATA_DIR", self.BASE_DIR / "data"))
        self.SECURITY_DIR = Path(os.getenv("SECURITY_DIR", self.BASE_DIR / "security"))
        self.LOG_DIR = Path(os.getenv("LOG_DIR", self.BASE_DIR / "logs"))

        # 确保目录存在
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.SECURITY_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def excel_file(self):
        return self.DATA_DIR / "ai_tools_900+.xlsx"

    @property
    def bookmark_file(self):
        return self.DATA_DIR / "bookmarks.xlsx"

# ================== 核心功能类 ==================

class HybridClassifier:
    """混合分类器（BERT + 规则引擎）"""
    def __init__(self):
        # 使用HuggingFace官方模型名称
        self.bert_model = pipeline(
            'feature-extraction',
            model="bert-base-multilingual-uncased",
            tokenizer="bert-base-multilingual-uncased"
        )

        self.rule_engine = RuleBasedClassifier()
        self.ml_model = None

    def predict(self, name, description):
        # 规则引擎优先
        rule_result = self.rule_engine.classify(name, description)
        if rule_result.confidence > 0.8:
            return rule_result.category

        # BERT模型预测
        text = f"{name}: {description}"[:500]
        bert_result = self.bert_model(text)
        top_category = max(bert_result, key=lambda x: x['score'])

        # 置信度检查
        if top_category['score'] > 0.7:
            return top_category['label']
        else:
            return self.ml_predict(name, description)

class RuleBasedClassifier:
    """增强型规则分类器"""

    def __init__(self):
        self.rules = CATEGORY_KEYWORDS

    def classify(self, name, desc):
        text = f"{name} {desc}".lower()
        scores = {}
        for cat, keywords in self.rules.items():
            scores[cat] = sum(1 for kw in keywords if kw in text)

        max_score = max(scores.values())
        if max_score > 0:
            best_cat = max(scores, key=scores.get)
            return ClassificationResult(
                category=best_cat,
                confidence=max_score / len(self.rules[best_cat])
            )
        return ClassificationResult('其他', 0.0)

# ================== 函数定义顺序 ==================
# 1. 基础工具函数
def standardize_country(raw_name, mapping):
    """返回简体中文国家名称"""
    raw = str(raw_name).strip().lower()
    for cn_name, aliases in mapping.items():
        if any(alias in raw for alias in aliases):
            return cn_name
    return '其他'  # 确保返回字符串类型

def preprocess_dates(df):
    """统一处理多种日期格式（增强版）"""
    if 'LastUpdated' not in df.columns:
        return df

    # 强制转换为字符串并处理空值
    df['LastUpdated'] = (
        df['LastUpdated']
        .astype(str)
        .replace('nan', '')  # 处理pandas的NaN字符串表示
        .replace('NaT', '')  # 处理时间空值
        .fillna('')  # 双重保险
    )

    # 移除时间部分（增强正则）
    df['LastUpdated'] = (
        df['LastUpdated']
        .str.replace(
            r'\s*\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\s*',  # 匹配所有时间格式
            '',
            regex=True
        )
    )

    # 统一分隔符为-
    df['LastUpdated'] = (
        df['LastUpdated']
        .str.replace(r'[/年月日]', '-', regex=True)  # 合并替换操作
        .str.replace(r'-+', '-', regex=True)  # 标准化分隔符
    )

    # 精确提取日期部分
    df['LastUpdated'] = df['LastUpdated'].str.extract(
        r'(\d{4}-\d{1,2}-\d{1,2})',
        expand=False
    )

    return df

def fill_missing_dates(df):
    """支持混合日期格式的智能解析"""
    df = preprocess_dates(df)

    # 定义常见日期格式（按优先级排序）
    date_formats = [
        '%Y-%m-%d',  # ISO格式
        '%Y/%m/%d',  # 斜杠格式
        '%Y%m%d',  # 紧凑格式
        '%Y年%m月%d日',  # 中文格式
        '%d-%b-%y'
    ]


    df['LastUpdated'] = pd.to_datetime(
        df['LastUpdated'],
        errors='coerce',  # 转换失败设为 NaT
        format='mixed',  # 自动检测格式（Pandas 2.0+ 特性）
        dayfirst=False  # 不使用日优先格式
    )

    # 填充缺失日期为当前日期（无时区）
    now = pd.Timestamp.now().floor('D')
    df['LastUpdated'] = df['LastUpdated'].fillna(now)

    return df

def clean_country_names(df):
    """国家名称标准化"""
    country_mapping = COUNTRY_MAPPING
    df['Country'] = df['Country'].apply(lambda x: standardize_country(x, country_mapping))
    return df

def validate_url(url):
    """URL标准化处理"""
    if pd.isna(url) or url in ['', '无', 'none']:
        return ''
    url = str(url).strip()
    if not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url

def validate_urls(df):
    """批量URL标准化处理"""
    df['URL'] = df['URL'].apply(validate_url)
    return df

# 3.业务逻辑函数
def calculate_popularity(df):
    """直接读取原始流行度值"""
    df['Popularity'] = (
        df['Popularity']
        .astype(int)
        .clip(lower=EnhancedConfig.MIN_POPULARITY, upper=1000)
    )
    return df

def sanitize_text(text):
    """文本安全处理"""
    return text.strip().replace('\x00', '')  # 移除空字符等特殊符号

def self_healing_data(df):
    """数据自愈功能（自动修复常见问题）"""
    # 修复名称中的多余空格
    df['Name'] = df['Name'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 自动识别开源字段
    df['Open Source'] = df.apply(
        lambda x: '是' if ('开源' in str(x['Description'])) else x['Open Source'],
        axis=1
    )
    return df
    print(df['LastUpdated'].dtype)

def calculate_trend(row):
    """增强趋势计算（结合时间因素）"""
    base = row['Popularity']
    days_since_update = (datetime.datetime.now() - row['LastUpdated']).days

    # 根据更新时间和流行度综合判断
    if days_since_update <= 7:  # 一周内更新
        if base > 700:
            return '🚀 爆火新星'
        elif base >= 500:
            return '✨ 近期热门'
    else:
        if base > 800:
            return '🏆 长期热门'

    if base >= 600:
        return '📈 上升趋势'
    elif base >= 400:
        return '🆗 保持稳定'
    else:
        return '⏳ 潜力待挖'

def validate_opensource(value):
    """开源字段标准化"""
    value = str(value).strip().lower()
    return '是' if value in ['是', 'yes', 'y', 'true', '开源'] else '否'

# 4. 分类相关函数 =
def classify_tool(name, description):
    """升级后的分类入口"""
    classifier = HybridClassifier()
    return classifier.predict(name, description)

# 5. 数据质量检查
def validate_row(row):
    """验证单行数据有效性"""
    errors = []

    # 检查必填字段
    if pd.isna(row['Name']):
        errors.append("名称不能为空")
    if pd.isna(row['Category']):
        errors.append("分类不能为空")

    return errors if errors else None

def data_quality_check(df):
    """日期质量验证"""
    errors = []

    # 检查日期是否全部解析成功
    if df['LastUpdated'].isna().any():
        bad_count = df['LastUpdated'].isna().sum()
        errors.append(f"{bad_count} 条日期解析失败")

    # 检查日期范围合理性
    latest_date = df['LastUpdated'].max()
    if latest_date > pd.Timestamp.now() + pd.DateOffset(years=1):
        errors.append("存在未来超过1年的日期")

    return errors

@st.cache_data(ttl=3600, show_spinner="加载数据中...")
def load_data(path):
    try:
        df = pd.read_excel(path, sheet_name="Tools")
        df.columns = df.columns.str.title()  # 新增列名标准化
        df = df.dropna(subset=['Name', 'Url'])  # 注意列名改为首字母大写
        return preprocess_data(df)
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        return pd.DataFrame()

def preprocess_data(df):
    """数据预处理流水线"""
    # 确保列名统一为首字母大写
    required_columns = ['Name', 'Category', 'Country', 'Url', 'Popularity', 'Lastupdated']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # 处理国家列
    df['Country'] = df['Country'].apply(lambda x: str(x).title())  # 统一国家列格式
    df['Country'] = df['Country'].apply(lambda x: x if x in COUNTRY_FLAGS else '其他')

    # 处理日期列
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'], errors='coerce').fillna(pd.Timestamp.now())

    # 处理流行度
    df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')
    df['Popularity'] = df['Popularity'].fillna(50).astype(int).clip(0, 1000)

    # 处理分类列
    if 'Category' not in df.columns:
        df['Category'] = '未分类'
    df['Category'] = df['Category'].str.title().fillna('未分类')

    return df

def load_tool_data(excel_file):
    """加载工具数据（修正版）"""
    categories = {}
    seen = set()  # 添加初始化
    try:
        # 定义预期的列名列表
        expected_columns = [
            'Name', 'Category', 'Country', 'Company', 'Description',
            'URL', 'Open Source', 'Popularity', 'LastUpdated'
        ]
        # ========== 读取数据 ==========
        # 读取时限制列并验证
        df = pd.read_excel(
            excel_file,
            sheet_name="Tools",
            usecols=expected_columns, # 关键修改：强制使用预期列
            header=0)
        # 添加分类字段处理
        if 'Category' not in df.columns:
            df['Category'] = '未分类'
        else:
            df['Category'] = df['Category'].str.strip().fillna('未分类')
        # 新增：填充所有文本列的空值为空字符串
        text_columns = ['Name', 'Category', 'Country', 'Company', 'Description', 'URL']
        df[text_columns] = df[text_columns].fillna('')

        # 确保列存在性
        required_columns = ['Name', 'Popularity', 'LastUpdated']
        # 验证关键列数据完整性
        missing_data = df[required_columns].isna().any(axis=1)
        if missing_data.any():
            st.error(f"发现 {missing_data.sum()} 行缺失关键数据")
            st.dataframe(df[missing_data][required_columns])
            df = df.dropna(subset=required_columns)

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"❌ 缺少必要列: {', '.join(missing_cols)}")
            return {}

        df = df.dropna(subset=['Popularity', 'LastUpdated'])
        # 如果成功读取数据，继续进行处理
        if df.empty:
            st.warning("⚠️ 工具数据为空，请检查Excel文件")
            return {}

        # 1. 处理缺失字段
        required_columns = {
            'Popularity': 0,
            'LastUpdated': pd.Timestamp.now().floor('D'),
            'Category': None  # 特殊处理分类字段
        }

        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val if col != 'Category' else pd.NA
                if col != 'Category':  # 分类字段需要特殊处理
                    st.warning(f"⚠️ 检测到缺失字段 '{col}'，已自动生成默认值")

        # 应用验证
        validation_results = df.apply(validate_row, axis=1)
        if any(validation_results):
            st.error("发现数据验证错误：")
            for idx, errors in enumerate(validation_results):
                if errors:
                    st.write(f"第 {idx + 2} 行错误：{', '.join(errors)}")
            return {}
            # 预处理阶段添加日期清洗
        df = (df
              .pipe(clean_country_names)
              .pipe(fill_missing_dates)
              .pipe(validate_urls)
              .pipe(calculate_popularity)  # 使用简化后的计算
              .pipe(self_healing_data)
              )
        # 在 load_tool_data 中调用
        errors = data_quality_check(df)
        if errors:
            st.error("⚠️ 数据质量问题: " + ", ".join(errors))
            st.dataframe(df[df['LastUpdated'].isna()])  # 展示错误数据

        # 添加日期格式验证
        invalid_dates = df[df['LastUpdated'].isna()]
        if not invalid_dates.empty:
            st.error(f"发现 {len(invalid_dates)} 条无效日期记录")
            st.dataframe(invalid_dates[['Name', 'LastUpdated']])
            df = df.dropna(subset=['LastUpdated'])


        # 3. 自动分类补全（原功能增强）
        df['Category'] = df.apply(
            lambda row: classify_tool(row['Name'], row['Description'])
            if pd.isna(row['Category']) else row['Category'],
            axis=1
        )
        # 添加类型检查
        if not pd.api.types.is_datetime64_ns_dtype(df['LastUpdated']):
            st.error("日期列类型转换失败，请检查数据格式")
            return {}
        # ========== 数据标准化处理 ==========
        for _, row in df.iterrows():
            # 字段有效性检查（增强版）
            if any([
                pd.isna(row['Name']),
                pd.isna(row['Category']),
                pd.isna(row['URL'])
            ]):
                continue

            # 数据标准化处理
            item = {
                'name': str(row['Name']).strip(),
                'category': str(row['Category']).strip(),
                'url': validate_url(row['URL']),
                'description': sanitize_text(row.get('Description')),
                'country':standardize_country(str(row.get('Country', '')),mapping=COUNTRY_MAPPING),
                'open_source': validate_opensource(row.get('Open Source', '否')),
                'company': str(row.get('Company', '')).strip(),
                'popularity': int(row.get('Popularity', 0)),
                'last_updated': row['LastUpdated'].to_pydatetime().replace(tzinfo=None),  # 移除时区信息
                'trend': calculate_trend(row)

            }

            # 防止重复条目
            key = (item['name'], item['category'])
            if key in seen:
                continue
            seen.add(key)

            # 有效性最终验证
            if not all([item['name'], item['category'], item['url']]):
                continue

            # 分类存储
            categories.setdefault(item['category'], []).append(item)
            item['last_updated'] = row['LastUpdated'].to_pydatetime().replace(tzinfo=None)

        return categories

    except FileNotFoundError:
        st.error("❌ 数据文件未找到，请检查文件路径")
        return {}
    except Exception as e:
        st.error(f"🚨 数据加载错误：{str(e)}")
        return {}

def load_tutorial_data(excel_file):
    """加载教程数据"""
    try:
        df = pd.read_excel(excel_file, sheet_name='Tutorials')
        tutorials = {}
        for _, row in df.iterrows():
            tool_name = row['RelatedTool']
            tutorial = {
                'id': row['TutorialID'],
                'title': row['Title'],
                'url': row['URL'],
                'type': row['Type'],
                'difficulty': row['DifficultyLevel'],
                'duration': row.get('Duration', ''),
                'rating': row.get('Rating', None),
                'language': row.get('Language', '中文'),
                'tags': [t.strip() for t in str(row.get('Tags', '')).split(',')],
                'version': row.get('VersionCompatible', ''),
                'author': row.get('Author', '')
            }
            tutorials.setdefault(tool_name, []).append(tutorial)
        return tutorials
    except Exception as e:
        st.error(f"教程数据加载失败: {str(e)}")
        return {}

def save_all_data(excel_file, tools_data, tutorials_data):
    """保存所有数据到Excel"""
    try:
        tools_rows = []
        for category, items in tools_data.items():
            for item in items:
                tools_rows.append({
                    'Name': item['name'],
                    'Category': category,
                    'Country': item['country'],
                    'Company': item.get('company', ''),
                    'Description': item['description'],
                    'URL': item['url'],
                    'Open Source': item['open_source']
                })
        tools_df = pd.DataFrame(tools_rows)

        tutorials_rows = []
        for tool_name, tutorials in tutorials_data.items():
            for tut in tutorials:
                tutorials_rows.append({
                    'TutorialID': tut['id'],
                    'RelatedTool': tool_name,
                    'Title': tut['title'],
                    'URL': tut['url'],
                    'Type': tut['type'],
                    'DifficultyLevel': tut['difficulty'],
                    'Duration': tut.get('duration', ''),
                    'Rating': tut.get('rating', None),
                    'Language': tut.get('language', '中文'),
                    'Tags': ', '.join(tut.get('tags', [])),
                    'VersionCompatible': tut.get('version', ''),
                    'Author': tut.get('author', '')
                })
        tutorials_df = pd.DataFrame(tutorials_rows)

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            tools_df.to_excel(writer, sheet_name='Tools', index=False)
            tutorials_df.to_excel(writer, sheet_name='Tutorials', index=False)
        return True
    except Exception as e:
        st.error(f"保存失败：{str(e)}")
        return False
# 7. UI组件函数
def render_url(url):
    """根据URL生成超链接"""
    if url and url != '无':
        return f'<a href="{url}" class="link-btn" target="_blank">{url}</a>'
    else:
        return "<span>无可用链接</span>"

def render_tool_card(item):
    if not isinstance(item, dict):
        st.error(f"非法数据项类型: {type(item)}")
        return

    country = item.get('country', 'Unknown')
    flag = COUNTRY_FLAGS.get(country, '🌐')
    badge = "🟢 开源" if item.get('open_source') == '是' else "🔴 闭源"
    company_info = f" - 公司：{item.get('company', '')}" if item.get('company') else ""

    url = item.get('url', None)

    popularity = item.get('popularity', 500)

    trend = item.get('trend', '')

    html = f"""
     <div class="item-card">
         <div style="display:flex; align-items:start; gap:12px">
             <div style="flex:1">
                 <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px">
                     <div style="font-size:1.1rem; font-weight:600">{item['name']}</div>
                     <div style="font-size:1.2em">{flag}</div>
                     <div style="margin-left:auto; display:flex; align-items:center; gap:6px">
                         <span style="font-size:0.8em; background:{'#e8f5e9' if item['open_source'] == '是' else '#ffebee'}; 
                             color:{'#2e7d32' if item['open_source'] == '是' else '#c62828'}; 
                             padding:2px 8px; border-radius:12px">
                             {'开源' if item['open_source'] == '是' else '闭源'}
                         </span>
                     </div>
                 </div>
                 <div style="display:flex; align-items:center; gap:8px; margin-top:8px">
                     <a href="{url}" target="_blank" class="link-btn" style="font-size:0.9em">
                         🔗 访问官网
                     </a>
                     <span class="popularity-badge">
                         🔥 {popularity} · {trend}
                     </span>
                 </div>
             </div>
         </div>
         <div class="tooltip">
            <h4>公司 - {item['company']}</h4>
            <h4>国家 - {item['country']}</h4>
            <div class="tooltip-content">
                {item.get('description', '暂无详细描述')}
                <div class="features">
                    <hr style="margin:8px 0">
                    网址 - {item.get('url', '无可用链接')}
                   <!-- 主要功能：
                    {item.get('features', '功能信息待补充')} -->
                </div>
            </div>
        </div>
    </div>
 </div>
"""
    st.markdown(html, unsafe_allow_html=True)

def render_tutorial_search(tutorials_data, all_tools):
    """增强版教程搜索组件"""
    with st.expander("🔍 高级教程搜索", expanded=False):
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            search_term = st.text_input("输入工具名称或关键词")
        with col2:
            difficulty = st.selectbox("难度等级", ["全部", "初级", "中级", "高级"])
        with col3:
            tutorial_type = st.selectbox("教程类型", ["全部", "视频教程", "文档指南", "实战案例", "社区讨论"])

        # 智能搜索算法
        results = []
        for tool_name, tutorials in tutorials_data.items():
            tool_match = any([
                not search_term,
                search_term.lower() in tool_name.lower(),
                search_term.lower() in ' '.join(all_tools.get(tool_name, {}).get('tags', [])).lower()
            ])

            if tool_match:
                for tutorial in tutorials:
                    content_match = any([
                        not search_term,
                        search_term.lower() in tutorial['title'].lower(),
                        search_term.lower() in ' '.join(tutorial['tags']).lower()
                    ])

                    type_match = (tutorial_type == "全部") or (tutorial['type'] == tutorial_type)
                    difficulty_match = (difficulty == "全部") or (tutorial['difficulty'] == difficulty)

                    if content_match and type_match and difficulty_match:
                        results.append({
                            'tool': tool_name,
                            **tutorial
                        })

        # 结果展示
        if results:
            st.success(f"🎉 找到 {len(results)} 个相关教程")
            for result in results:
                badge_class = {
                    '视频教程': 'video-badge',
                    '文档指南': 'doc-badge',
                    '实战案例': 'case-badge',
                    '社区讨论': 'community-badge'
                }.get(result['type'], '')

                with st.container(border=True):
                    cols = st.columns([3, 1, 1, 1])
                    cols[0].markdown(
                        f"**[{result['title']}]({result['url']})**  \n"
                        f"<span class='tutorial-badge {badge_class}'>{result['type']}</span> "
                        f"<span style='font-size:0.9em;color:#666'>{result['tool']}</span>",
                        unsafe_allow_html=True
                    )
                    cols[1].write(f"**难度**  \n{result['difficulty']}")


                    if result.get('tags'):
                        tags_str = " ".join([f"#{t}" for t in result['tags']])
                        st.caption(f"标签：{tags_str}")
                    else:
                        st.info("""
                                    📘 推荐学习路径：
                                    1. 访问[AI学习中心 - 吴恩达《机器学习》课程](https://www.coursera.org/learn/machine-learning)
                                    2. 加入我们的[开发者社区 - TensorFlow论坛](https://discuss.tensorflow.org/)
                                    3. 查看[最新工具动态 - Hugging Face博客](https://huggingface.co/blog)
                                    4. 访问[DeepLearning.AI课程平台](https://www.deelearning.ai)
                                    5. 加入[PyTorch开发者讨论区](https://discuss.pytorch.org)
                                    6. 跟踪[Google AI最新动态](https://ai.google/blog)
                                    7. 学习[李沐《动手学深度学习》](https://zh.d2l.ai)
                                    8. 参与[ApacheCN中文社区](https://www.apachecn.org)
                                    9. 关注[机器之心AI新闻](https://www.jiqizhixin.com)
                                    """)

def load_model():
    """加载已有模型"""
    model_path = PathConfig().SECURITY_DIR / "text_classifier.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

# 10. 搜索引擎集成
def real_time_search(query):
    """使用Serper API进行全网搜索"""
    url = "https://google.serper.dev/search"
    payload = {
        "q": f"{query} AI工具 教程 最新资讯",
        "gl": "cn",
        "hl": "zh-cn",
        "num": 5
    }
    headers = {
        'X-API-KEY': EnhancedConfig.SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        results = response.json()
        return parse_serper_results(results)
    except Exception as e:
        st.error(f"搜索失败：{str(e)}")
        return []

def parse_serper_results(data):
    """解析搜索引擎结果"""
    parsed = []
    for result in data.get('organic', [])[:5]:
        parsed.append({
            'title': result.get('title'),
            'url': result.get('link'),
            'snippet': result.get('snippet'),
            'date': detect_date(result.get('snippet', ''))
        })
    return parsed

def detect_date(text):
    """从文本中提取日期"""
    try:
        dt = parse(text, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except:
        return "近期"

def refresh_tools():
    st.session_state.show_tools = True
    st.rerun()

# ================== 新增函数 ==================
def load_bookmark_data(excel_file):
    """加载收藏数据"""
    try:
        df = pd.read_excel(excel_file, sheet_name="Bookmarks")
        bookmarks = []
        # 添加列名兼容性处理
        column_mapping = {
            "标题": "title",
            "URL": "url",
            "分类": "category",
            "标签": "tags",
            "备注": "notes",
            "添加时间": "add_date"
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        for _, row in df.iterrows():
            bookmarks.append({
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "category": row.get("category", "未分类"),
                "tags": row.get("tags", "").split(",") if pd.notna(row.get("tags")) else [],
                "notes": row.get("notes", ""),
                "add_date": row.get("add_date", datetime.datetime.now().strftime("%Y-%m-%d"))
            })
        return bookmarks
    except Exception as e:
        st.error(f"收藏数据加载失败: {str(e)}")
        return []

def save_bookmark_data(excel_file, bookmarks):
    """保存收藏数据（修复版）"""
    try:
        # 转换数据时使用与加载时一致的列名
        df = pd.DataFrame([{
            "标题": b["title"],
            "URL": b["url"],
            "分类": b["category"],
            "标签": ",".join(b["tags"]),
            "备注": b["notes"],
            "添加时间": b["add_date"]
        } for b in bookmarks])

        # 使用正确的写入模式
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name="Bookmarks", index=False)
        return True
    except Exception as e:
        st.error(f"收藏保存失败: {str(e)}")
        return False

# 新增热点资讯获取函数
@st.cache_data(ttl=3600, show_spinner="获取热点资讯中...")
def fetch_hot_news(query="AI技术 人工智能 大模型 DeepSeek OpenAI ChatGPT"):
    try:
        params = {
            "q": query,
            "sortBy": "popularity",
            "pageSize": 5,
            "language": "zh",
            "apiKey": NEWS_API_KEY
        }

        response = requests.get(NEWS_API_ENDPOINT, params=params)
        data = response.json()

        articles = data.get('articles', [])
        processed = []
        for article in articles[:5]:  # 取前5条
            processed.append({
                "title": article.get('title', ''),
                "url": article.get('url', '#'),
                "source": article.get('source', {}).get('name', '未知来源'),
                "snippet": article.get('description', ''),
                "time": article.get('publishedAt', '')[:10]
            })
        return processed
    except Exception as e:
        st.error(f"热点资讯获取失败: {str(e)}")
        return [{
            "title": "AI技术最新突破：多模态大模型取得重大进展",
            "url": "#",
            "source": "虚拟新闻",
            "snippet": "近日，全球顶尖研究团队宣布在跨模态理解领域取得突破性进展...",
            "time": datetime.date.today().strftime("%Y-%m-%d")
        }]

# 11. 主程序
def main():
    st.set_page_config(
        page_title="AI工具导航中心",
        layout="wide",
        menu_items={
            'Get Help': 'https://example.com/help',
            'Report a bug': "mailto:1134593154@qq.com",
        }
    )

    # 标题优化
    st.markdown("""
        <h1 class="header gradient-text">🤖 AI工具导航中心</h1>
        <div style="text-align:center; margin-bottom:2rem">
            <div style="display:inline-block; background: #f0f4f8; padding:8px 20px; border-radius:20px">
                <span style="color:#666">数据更新：2025年2月</span>
                <span style="margin:0 10px">|</span>
                <span style="color:#666">版本：2.3 </span>
                <span style="margin:0 10px">|</span>
                <span style="color:#666">作者：刘晓伟</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ================== 数据加载 ==================
    path_config = PathConfig()
    excel_file = path_config.excel_file

    try:
        tools_data = load_tool_data(excel_file)
        tutorials_data = load_tutorial_data(excel_file)
        all_tools = {item['name']: item for cat in tools_data.values() for item in cat}
        st.session_state.tutorials_data = tutorials_data
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return


    # ================== 侧边栏优化 ==================
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin:1rem 0">
            <div style="font-size:1.2rem; color:#2c3e50; font-weight:600">🔍 导航面板</div>
            <div style="height:2px; background:linear-gradient(90deg, transparent 0%, #3498db 50%, transparent 100%); margin:0.5rem 0"></div>
        </div>
        """, unsafe_allow_html=True)

        # 分类选择改用选项卡
        selected_main = st.selectbox(
            "选择主分类",
            options=list(CATEGORY_STRUCTURE.keys()),
            index=0,
            format_func=lambda x: x.split(' ')[0]  # 仅显示图标和名称
        )

        # 子分类选择优化
        sub_categories = CATEGORY_STRUCTURE[selected_main]
        selected_sub = st.selectbox(
            f"选择{selected_main.split(' ')[0]}子类",
            options=sub_categories,
            index=0,
            help="请选择具体的工具子分类"
        )


        st.header("⭐ 我的知识宝库")

        # 加载收藏数据
        if 'bookmarks' not in st.session_state:
            st.session_state.bookmarks = load_bookmark_data(path_config.bookmark_file)

        # 搜索和筛选
        col1, col2 = st.columns([3, 2])
        search_term = col1.text_input("搜索收藏内容")
        filter_category = col2.selectbox("筛选分类", ["全部"] + list(BOOKMARK_CATEGORIES.keys()))

        # 展示收藏
        filtered = [
            b for b in st.session_state.bookmarks
            if (not search_term or search_term in b["title"]) and
               (filter_category == "全部" or b["category"] == filter_category)
        ]

        if filtered:
            for b in filtered:
                with st.expander(f"{b['title']} ({b['category']})", expanded=False):
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"🔗 [{b['url']}]({b['url']})")
                    cols[1].markdown(f"**添加时间**: {b['add_date']}")

                    if b["tags"]:
                        st.markdown(f"**标签**: {' '.join([f'🏷️{t}' for t in b['tags']])}")

                    if b["notes"]:
                        st.markdown(f"**备注**: {b['notes']}")

                    if st.button("删除", key=f"del_{b['url']}"):
                        st.session_state.bookmarks = [bm for bm in st.session_state.bookmarks
                                                      if bm['url'] != b['url']]
                        save_bookmark_data(path_config.bookmark_file, st.session_state.bookmarks)
                        st.rerun()
        else:
            st.info("暂未收藏任何内容，快去发现精彩资源吧！")
        # 筛选条件分组
        with st.expander("⚙️ 高级筛选", expanded=False):
            min_popularity = st.slider(
                "最低流行度", 0, 1000, 300,
                help="过滤流行度低于该值的工具"
            )

            country_filter = st.multiselect(
                "国家/地区筛选",
                options=list(COUNTRY_FLAGS.keys()),
                default=[],
                format_func=lambda x: f"{COUNTRY_FLAGS.get(x, '🌍')} {x}"
            )

            st.divider()
    # ================== 主内容区域 ==================
    try:
        # 显示当前分类标题
        st.subheader(f"{selected_main} - {selected_sub}")

        # 获取筛选后的数据
        filtered_items = [
            item for item in tools_data.get(selected_sub, [])
            if item['popularity'] >= min_popularity
               and (not country_filter or item['country'] in country_filter)
        ]

        cols = st.columns(3)
        stats_style = """
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;  # 新增水平居中
            transition: transform 0.2s ease;  # 添加悬停动画
            border: 1px solid #f0f0f0;  # 添加柔和边框
        """

        # 为所有列添加统一悬停效果
        hover_style = """
            <style>
                div[data-testid="stVerticalBlock"] > div > div > div:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
                }
            </style>
        """
        st.markdown(hover_style, unsafe_allow_html=True)

        with cols[0]:
            st.markdown(f"""
                <div style="{stats_style}">
                    <div style="color:#666; font-size:0.95rem; margin-bottom:8px">📦 当前分类工具数</div>
                    <div style="font-size:2rem; color:#2a3f5f; font-weight:800">{len(filtered_items)}</div>
                </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"""
                <div style="{stats_style}">
                    <div style="color:#666; font-size:0.95rem; margin-bottom:8px">📈 平均流行度</div>
                    <div style="font-size:2rem; color:#2a3f5f; font-weight:800">
                        {int(np.mean([i['popularity'] for i in filtered_items])) if filtered_items else 0}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            countries = [i['country'] for i in filtered_items]
            st.markdown(f"""
                <div style="{stats_style}">
                    <div style="color:#666; font-size:0.95rem; margin-bottom:8px">🌍 主要国家</div>
                    <div style="font-size:2rem; color:#2a3f5f; font-weight:800">
                        {max(set(countries), key=countries.count) if countries else "无"}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # 显示工具卡片
        if filtered_items:
            cols = st.columns(2)
            col_idx = 0
            for item in filtered_items:
                with cols[col_idx]:
                    render_tool_card(item)
                col_idx = 1 - col_idx  # 切换列
        else:
            st.markdown("""
              <div style="text-align:center; padding:3rem; background:white; border-radius:12px">
                  <div style="font-size:1.2rem; color:#666; margin-bottom:1rem">😞 未找到匹配工具</div>
                  <div style="color:#888">建议尝试调整筛选条件或查看其他分类</div>
              </div>
              """, unsafe_allow_html=True)

        # ================== 教程搜索区域 ==================
        with st.expander("📚本地工具+教程搜索栏", expanded=False):
            st.subheader("🔍 快速搜索")

            # 工具搜索功能模块
            def search_tools(search_term, search_field='name'):
                return [
                    item for cat in tools_data.values() for item in cat
                    if search_term.lower() in item[search_field].lower()
                ]

            # 双列布局改进版
            col_search1, col_search2 = st.columns(2)

            with col_search1:
                # 工具名称搜索
                search_term = st.text_input("输入工具名称关键词",
                                            placeholder="例如：图像识别",
                                            key="name_search")
                if search_term:
                    name_results = search_tools(search_term, 'name')
                    if name_results:
                        st.success(f"名称匹配：找到 {len(name_results)} 个工具")
                        for item in name_results:
                            render_tool_card(item)
                    else:
                        st.info("⚠️ 未找到名称匹配的工具")

            with col_search2:
                # 功能描述搜索
                search_term_function = st.text_input("输入功能关键词",
                                                     placeholder="例如：背景虚化",
                                                     key="func_search")
                if search_term_function:
                    func_results = search_tools(search_term_function, 'description')
                    if func_results:
                        st.success(f"功能匹配：找到 {len(func_results)} 个工具")
                        for item in func_results:
                            render_tool_card(item)
                    else:
                        st.info("⚠️ 未找到功能匹配的工具")

            # 教程搜索模块
            st.divider()
            st.subheader("🎓 手把手教程搜索")

            # 改进的教程搜索逻辑
            tutorial_search = st.text_input("输入AI工具名称查找教程",
                                            placeholder="例如：Photoshop AI",
                                            key='tutorial_search')

            if tutorial_search:
                matched_tutorials = []
                for tool_name, tutorial_list in tutorials_data.items():
                    # 处理不同的数据结构情况
                    if isinstance(tutorial_list, list) and len(tutorial_list) > 0:
                        if tutorial_search.lower() in tool_name.lower():
                            for tutorial in tutorial_list:
                                matched_tutorials.append((
                                    tool_name,
                                    tutorial.get('url', '#'),
                                    tutorial.get('source', '未知来源')
                                ))

                if matched_tutorials:
                    st.success(f"🔍 找到 {len(matched_tutorials)} 个相关教程")
                    for name, url, source in matched_tutorials:
                        # 使用streamlit原生样式
                        with st.container(border=True):
                            st.markdown(f"""
                            **{name}**  
                            📚 教程来源：{source}  
                            🔗 [点击查看完整教程]({url})
                            """)
                else:
                    # 优化后的学习资源推荐
                    st.info("""
                    📘 推荐学习路径：
                    1. [机器学习基础 - 吴恩达 Coursera 课程](https://www.coursera.org/learn/machine-learning)
                    2. [深度学习实战 - Fast.ai 课程](https://www.fast.ai)
                    3. [自然语言处理 - Hugging Face 教程](https://huggingface.co/learn)
                    4. [计算机视觉 - PyTorch 官方教程](https://pytorch.org/tutorials/)
                    5. 加入我们的[开发者社区 - TensorFlow论坛](https://discuss.tensorflow.org/)
                    6. 查看[最新工具动态 - Hugging Face博客](https://huggingface.co/blog)
                    7. 访问[DeepLearning.AI课程平台](https://www.deelearning.ai)
                    8. 加入[PyTorch开发者讨论区](https://discuss.pytorch.org)
                    9. 跟踪[Google AI最新动态](https://ai.google/blog)
                    10. 学习[李沐《动手学深度学习》](https://zh.d2l.ai)
                    11. 参与[ApacheCN中文社区](https://www.apachecn.org)
                    12. 关注[机器之心AI新闻](https://www.jiqizhixin.com)
                    """)
        with st.expander("🌐 全网AI工具搜索", expanded=False):
            search_query = st.text_input("输入搜索关键词（支持中英文）",
                                         key="web_search",
                                         help="搜索最新AI工具资讯和教程")

            if search_query:
                with st.spinner('正在搜索全网最新资讯...'):
                    # 执行实时搜索
                    results = real_time_search(search_query)

                if results:
                    st.success(f"找到 {len(results)} 条相关结果")
                    for idx, item in enumerate(results[:5], 1):  # 显示前3条结果
                        st.markdown(f"""
                                         <div class="item-card">
                                             <b>{idx}. {item['title']}</b>
                                             <div class="tooltip">
                                                 📅 {item['date']}
                                                 <span class="tooltiptext">
                                                     {item['snippet']}
                                                 </span>
                                             </div>
                                             <div style="margin-top:8px;">
                                                 <a href="{item['url']}" class="link-btn" target="_blank">查看详情</a>
                                             </div>
                                         </div>
                                         """, unsafe_allow_html=True)
                else:
                    st.info("""
                                     💡 未找到相关结果，建议：
                                     1. 尝试不同关键词组合
                                     2. 查看[热门工具榜单](#)
                                     3. 访问[AI新闻聚合站](#)
                                     """)
        st.markdown(STYLE, unsafe_allow_html=True)
        render_tutorial_search(tutorials_data, all_tools)
        # ================== 新增热点资讯公告栏 ==================
        with st.expander("📰 点击展开实时热点资讯", expanded=False):
            st.markdown("""
             <div class="news-ticker">
                 <div class="news-header">🔥 热点资讯</div>
                 <div class="news-marquee">
             """, unsafe_allow_html=True)

            hot_news = fetch_hot_news()
            for news in hot_news:
                st.markdown(f"""
                     <div class="news-card">
                         <div class="news-title">{news['title']}</div>
                         <div class="news-source">来源：{news['source']} • {news['time']}</div>
                         <div class="news-snippet">{news['snippet']}</div>
                         <a href="{news['url']}" target="_blank" class="link-btn" style="font-size:0.8em">阅读全文 →</a>
                     </div>
                 """, unsafe_allow_html=True)
    except KeyError as e:
        st.error(f"分类数据错误: {str(e)}")

if __name__ == '__main__':
    main()