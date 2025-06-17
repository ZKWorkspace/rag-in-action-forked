import sqlite3
import os

# 数据库文件路径
db_path = 'example.db'

# 检查并删除已存在的数据库文件
if os.path.exists(db_path):
    print(f"发现已存在的数据库文件: {db_path}")
    try:
        os.remove(db_path)
        print(f"已删除旧数据库文件: {db_path}")
    except Exception as e:
        print(f"删除数据库文件时出错: {e}")
        exit(1)

# 创建新的数据库连接
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print(f"正在创建新数据库: {db_path}")

cursor.execute('''
CREATE TABLE game_scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_name TEXT NOT NULL,
    description TEXT,
    difficulty_level INTEGER,
    boss_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''')

cursor.execute('''
INSERT INTO game_scenes (scene_name, description, difficulty_level, boss_name)
VALUES 
    ('花果山', '悟空的出生地，山清水秀，仙气缭绕', 2, '六耳猕猴'),
    ('水帘洞', '花果山中的洞穴，悟空的老家', 1, NULL),
    ('火焰山', '炙热难耐的火山地带，充满岩浆与烈焰', 4, '牛魔王'),
    ('龙宫', '东海龙王的宫殿，水下奇景', 3, '敖广'),
    ('灵山', '如来佛祖居住的圣地，佛光普照', 5, '如来佛祖');
''')

# 提交更改
conn.commit()

# 验证数据插入
cursor.execute('SELECT * FROM game_scenes')
results = cursor.fetchall()
print(f"\n成功插入 {len(results)} 条记录:")
for row in results:
    print(f"ID: {row[0]}, 场景: {row[1]}, 描述：{row[2]}, 难度: {row[3]}, 老怪：{row[4]}")

# 关闭连接
conn.close()
print(f"\n数据库文件已创建完成: {db_path}")