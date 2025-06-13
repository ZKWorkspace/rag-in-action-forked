from langchain_community.document_loaders import JSONLoader
print("=== JSONLoader 加载结果 ===")
print("1. 主角信息：")
main_loader = JSONLoader(
    file_path="90-文档-Data/灭神纪/人物角色.json",
    jq_schema='.mainCharacter | "姓名：" + .name + "，背景：" + .backstory',
    text_content=True
)
main_char = main_loader.load()
print(main_char)
print("\n2. 支持角色信息：")
support_loader = JSONLoader(
    file_path="90-文档-Data/灭神纪/人物角色.json",
    jq_schema='.supportCharacters[] | "姓名：" + .name + "，背景：" + .background',
    text_content=True
)
support_chars = support_loader.load()
print(support_chars)
print("\n3. 游戏特点")
import re
def game_features_meta_gen(record: str, metadata: dict) -> dict:
    # 只处理元数据，不处理内容
    # record here is a string formatted by jq_schema:
    # '背景设定: 基于西游记神话背景，融合虚构的玄幻世界\n战斗系统：独特的变身战斗系统，结合高难度动作玩法\n进度模式：技能树发展系统，解锁新的战斗技能和变身形态\n剧情模式：开放式地图探索，丰富的支线任务和隐藏剧情\n画面渲染：采用虚幻引擎5, 实现高质量画面表现和实时光线追踪'
    pattern = r'^([^:：]+)[:：]'
    keys = [re.match(pattern, line).group(1).strip() for line in record.split("\n")]
    metadata["feature_keys"] = keys
    return metadata

features_loader = JSONLoader(
    file_path="90-文档-Data/灭神纪/人物角色.json",
    jq_schema='.gameFeatures | "背景设定: " + (.worldSetting // "") + "\n战斗系统：" + (.combatSystem // "") + "\n进度模式：" + (.progression // "") + "\n剧情模式：" + (.exploration // "") + "\n画面渲染：" + (.graphics // "")',
    text_content=True, # Note: text_content=True makes param record of metadata_func as str type 
    metadata_func=game_features_meta_gen
)
features = features_loader.load()
# print(features)
for feature in features:
    print("内  容：")
    print(feature.page_content)  # 将显示格式化后的游戏特性
    print("\n元数据：")
    print(feature.metadata)  # 将显示其他元数据
    print("-" * 100)
# 3. 游戏特点
# 内  容：
# 背景设定: 基于西游记神话背景，融合虚构的玄幻世界
# 战斗系统：独特的变身战斗系统，结合高难度动作玩法
# 进度模式：技能树发展系统，解锁新的战斗技能和变身形态
# 剧情模式：开放式地图探索，丰富的支线任务和隐藏剧情
# 画面渲染：采用虚幻引擎5, 实现高质量画面表现和实时光线追踪

# 元数据：
# {
#   "source": "/mnt/c/Workspace/github/rag-in-action-forked/90-文档-Data/灭神纪/人物角色.json",
#   "seq_num": 1,
#   "feature_keys": [
#     "背景设定",
#     "战斗系统",
#     "进度模式",
#     "剧情模式",
#     "画面渲染"
#   ]
# }