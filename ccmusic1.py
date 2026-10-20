import os

BASE = r"G:\pyyyyy\cxo\5.音乐流派数据库"

print(f"BASE 存在: {os.path.exists(BASE)}")
print(f"\nBASE 下的内容:")
for item in os.listdir(BASE):
    p = os.path.join(BASE, item)
    if os.path.isdir(p):
        print(f"  [目录] {item}")
    else:
        size_kb = os.path.getsize(p) / 1024
        print(f"  [文件] {item} ({size_kb:.1f} KB)")

# 递归统计所有文件类型
print(f"\n=== 递归统计 ===")
exts = {}
total_size = 0
total_files = 0
for root, dirs, files in os.walk(BASE):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        exts[ext] = exts.get(ext, 0) + 1
        total_size += os.path.getsize(os.path.join(root, f))
        total_files += 1

print(f"总文件数: {total_files}")
print(f"总大小: {total_size / 1024 / 1024:.2f} MB")
print(f"扩展名统计: {exts}")

# 找所有 CSV / Excel / txt 文件
print(f"\n=== 标注文件位置 ===")
for root, dirs, files in os.walk(BASE):
    for f in files:
        if f.lower().endswith(('.csv', '.xlsx', '.xls', '.txt', '.json')):
            p = os.path.join(root, f)
            size_kb = os.path.getsize(p) / 1024
            print(f"  {p} ({size_kb:.1f} KB)")

# 找所有音频文件
print(f"\n=== 音频文件 ===")
audio_count = 0
for root, dirs, files in os.walk(BASE):
    for f in files:
        if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            audio_count += 1
            if audio_count <= 5:
                print(f"  {os.path.join(root, f)}")
print(f"音频文件总数: {audio_count}")