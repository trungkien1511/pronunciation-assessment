import os
import glob
import re
import shutil

root_dir = r"d:\test"

# 1. Đi dọc qua các file .py
py_files = glob.glob(os.path.join(root_dir, "**", "*.py"), recursive=True)
py_files.extend(glob.glob(os.path.join(root_dir, "*.md")))

for file_path in py_files:
    if "refactor.py" in file_path or ".venv" in file_path or "_deprecated" in file_path:
        continue
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    original = content
        
    # Thay thế import phoneme_assessment -> src
    content = content.replace("phoneme_assessment", "src")
    
    # Thay thế r"d:\test..." thành biến tương đối
    # Ở app.py
    if os.path.basename(file_path) == "app.py":
        content = content.replace(r'r"d:\test\l2arctic_release_v5.0\ABA\wav\arctic_a0001.wav"', '"l2arctic_release_v5.0/ABA/wav/arctic_a0001.wav"')
        content = content.replace(r'r"d:\test\wav2vec2-l2arctic_final"', '"wav2vec2-l2arctic_final"')
        content = content.replace(r'"d:/test/wav2vec2-l2arctic_final"', '"wav2vec2-l2arctic_final"')
        
    # Trong các scripts 
    elif "scripts" in file_path:
        content = content.replace(r'r"d:\test"', "os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))")
        content = content.replace(r'"d:/test/l2arctic_release_v5.0"', 'os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "l2arctic_release_v5.0"))')
        # fix build_dataset path
        content = content.replace(r'r"d:\test\l2arctic_release_v5.0"', "os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'l2arctic_release_v5.0')")
        content = content.replace(r'r"d:\test\train_metadata.json"', "os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'train_metadata.json')")
        
    # Trong các file lõi
    elif "model.py" in file_path or "dataset.py" in file_path or "metrics.py" in file_path or "inference.py" in file_path:
        # replace r"d:\test\..." string direct assignments
        content = re.sub(r'r?"d:\\test\\(.*?)"', r'"\1"', content)
        content = re.sub(r'r?"d:/test/(.*?)"', r'"\1"', content)
        
    if "README.md" in file_path or "task.md" in file_path:
        content = content.replace("phoneme_assessment", "src")
        
    if content != original:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated paths in {os.path.basename(file_path)}")

# 2. Đổi tên thư mục phoneme_assessment thành src
old_dir = os.path.join(root_dir, "phoneme_assessment")
new_dir = os.path.join(root_dir, "src")
if os.path.exists(old_dir):
    os.rename(old_dir, new_dir)
    print("Renamed phoneme_assessment to src")

print("Refactoring complete.")
