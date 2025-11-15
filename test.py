import os
import re
from pathlib import Path


def rename_factor_files():
    """
    将指定目录下的所有文件重命名为 gtja_alpha_i 格式
    例如：alpha107 -> gtja_alpha_107
    """
    # 目标目录
    target_dir = r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\factor_results_GTJA_100-191\factor_results_100_191_TW50'


    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"错误：目录不存在 - {target_dir}")
        return
    
    # 获取目录下所有文件
    files = os.listdir(target_dir)
    
    # 用于匹配 alpha 后跟数字的模式
    pattern = re.compile(r'alpha(\d+)', re.IGNORECASE)
    
    renamed_count = 0
    
    for filename in files:
        file_path = os.path.join(target_dir, filename)
        
        # 跳过目录，只处理文件
        if os.path.isdir(file_path):
            continue
        
        # 尝试从文件名中提取数字后缀
        match = pattern.search(filename)
        
        if match:
            # 提取数字部分
            number = match.group(1)
            # 获取文件扩展名
            file_ext = os.path.splitext(filename)[1]
            # 构建新文件名
            new_filename = f"gtja_alpha_{number}{file_ext}"
            new_file_path = os.path.join(target_dir, new_filename)
            
            # 检查新文件名是否已存在
            if os.path.exists(new_file_path) and new_file_path != file_path:
                print(f"警告：目标文件已存在，跳过 - {filename} -> {new_filename}")
                continue
            
            # 重命名文件
            try:
                os.rename(file_path, new_file_path)
                print(f"重命名成功：{filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"错误：重命名失败 - {filename}: {e}")
        else:
            print(f"跳过：文件名格式不匹配 - {filename}")
    
    print(f"\n完成！共重命名 {renamed_count} 个文件")


if __name__ == "__main__":
    rename_factor_files()

