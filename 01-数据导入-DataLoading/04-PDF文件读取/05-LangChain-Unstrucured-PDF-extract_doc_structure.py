import os
import pdf2image
from PIL import Image, ImageDraw, ImageFont
from langchain_unstructured import UnstructuredLoader
from os.path import splitext  # 添加splitext导入

file_path = ("90-文档-Data/山西文旅/云冈石窟-en.pdf")

# 获取文件名（不带扩展名）
file_name = splitext(os.path.basename(file_path))[0]
print(f"处理文件: {file_name}")

# 创建输出目录
output_dir = f"{file_name}-vis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = pdf2image.convert_from_path(file_path)
for i, image in enumerate(images):
    image.save(f'{output_dir}/page_{i+1}.png')

loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",
    # partition_via_api=True,
    # coordinates=True,
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

def extract_basic_structure(docs):
    """基础结构化提取:按文档类型组织内容"""
    # 定义类别映射
    category_map = {
        'Title': 'title',
        'NarrativeText': 'text', 
        'Image': 'image',
        'Table': 'table',
        'Footer': 'footer',
        'Header': 'header'
    }
    
    # 初始化结构字典
    structure = {cat: [] for cat in category_map.values()}
    structure['metadata'] = [] # 额外添加metadata类别
    
    # 遍历文档并分类
    for doc in docs:
        category = doc.metadata.get('category', 'Unknown')
        content = {
            'content': doc.page_content,
            'page': doc.metadata.get('page_number'),
            'coordinates': doc.metadata.get('coordinates')
        }
        
        target_category = category_map.get(category)
        if target_category:
            structure[target_category].append(content)

    return structure

# 使用示例
structure = extract_basic_structure(docs)

# 输出第2页的标题
print("第2页标题:")
for title in [t for t in structure['title'] if t['page'] == 2]:
    print(f"- {title['content']}")

CATEGORY_COLORS = {
    'Title': (255, 0, 0),          # 红色
    'NarrativeText': (0, 255, 0),  # 绿色
    'Image': (0, 0, 255),          # 蓝色
    'Table': (255, 255, 0),        # 黄色
    'Footer': (255, 0, 255),       # 紫色
    'Header': (0, 255, 255),       # 青色
    'Unknown': (128, 128, 128)     # 灰色
}
FONT_SIZE = 28
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
def draw_bbox(img_path, element):
    pos = element['position']
    category = element['type'] if element['type'] in CATEGORY_COLORS else 'Unknown'
    color = CATEGORY_COLORS.get(category)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [(pos['x1'], pos['y1']), (pos['x2'], pos['y2'])],
        outline=color,
        width=5
    )
    draw.text(
        (pos['x1'], pos['y1'] - FONT_SIZE),
        category,
        fill=color,
        font=FONT,
    )
    img.save(img_path)

def analyze_layout(docs):
    """分析文档的版面布局结构"""
    layout_analysis = {}
    
    for doc in docs:
        page = doc.metadata.get('page_number')
        coords = doc.metadata.get('coordinates', {})
        
        # 初始化页面信息
        if page not in layout_analysis:
            layout_analysis[page] = {
                'elements': [],
                'dimensions': {
                    'width': coords.get('layout_width', 0),
                    'height': coords.get('layout_height', 0)
                }
            }
        
        # 获取元素位置信息
        points = coords.get('points', [])
        if points:
            # 只需要左上和右下坐标点
            (x1, y1), (_, _), (x2, y2), _ = points
            
            # 构建元素信息
            element = {
                'type': doc.metadata.get('category', 'Unknown'),
                'content': (doc.page_content[:50] + '...') if len(doc.page_content) > 50 else doc.page_content,
                'position': {
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            }
            layout_analysis[page]['elements'].append(element)
            # Draw bbox
            draw_bbox(f"{output_dir}/page_{page}.png", element)
    return layout_analysis

# 使用示例
layout = analyze_layout(docs)

# 分析第一页的布局
print("第1页布局分析:")
if 1 in layout:
    page = layout[1]
    print(f"页面尺寸: {page['dimensions']['width']} x {page['dimensions']['height']}")
    print("\n元素分布:")
    
    # 按垂直位置排序显示元素
    for elem in sorted(page['elements'], key=lambda x: x['position']['y1']):
        print(f"\n类型: {elem['type']}")
        print(f"位置: ({elem['position']['x1']:.0f}, {elem['position']['y1']:.0f})")
        print(f"尺寸: {elem['position']['width']:.0f} x {elem['position']['height']:.0f}")
        print(f"内容: {elem['content']}")

# 寻找父子关系
cave6_docs = []
parent_id = -1
for doc in docs:
    if doc.metadata["category"] == "Title" and "Cave 6" in doc.page_content:
        parent_id = doc.metadata["element_id"]
    if doc.metadata.get("parent_id") == parent_id:
        cave6_docs.append(doc)

for doc in cave6_docs:
    print(doc.page_content)

external_docs = [] # 创建列表来存储外部链接的子文档
parent_id = -1 # 初始化父ID为-1
for doc in docs:
    # 检查文档是否为标题类型且内容包含"External links"
    if doc.metadata["category"] == "Title" and "External links" in doc.page_content:
        parent_id = doc.metadata["element_id"]
        external_docs.append(doc)
    # 检查文档的parent_id是否匹配我们找到的标题ID
    if doc.metadata.get("parent_id") == parent_id:       
        external_docs.append(doc) # 将属于这个标题的子文档都添加到结果列表中
for doc in external_docs:
    print(doc.page_content)


# def find_tables_and_titles(docs):
#   results = []
#   for doc in docs:
#     # 检查文档是否为表格类型
#     if doc.metadata.get("category") == "Table":
#       table = doc
#       parent_id = doc.metadata.get("parent_id")
#       # 查找表格对应的标题文档(parent_id匹配element_id)
#       title = next((doc for doc in docs if doc.metadata.get("element_id") == parent_id), None)
#       if title:
#         results.append({"table": table.page_content, "title": title.page_content})
#   return results

# results = find_tables_and_titles(cave6_docs)
# if results:
#   for result in results:
#     print("找到的表格和标题:")
#     print(f"标题: {result['title']}")
#     print(f"表格: {result['table']}")
# else:
#   print("未找到任何表格和标题")