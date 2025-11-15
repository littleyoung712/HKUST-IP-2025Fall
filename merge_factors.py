import pandas as pd
import os
import glob
from tqdm import tqdm

def merge_factors_by_market(output_folder, market_name, score_column='score'):
    """
    合并指定市场的所有股票因子数据
    
    Parameters:
    -----------
    output_folder : str
        output_data文件夹路径
    market_name : str
        市场名称（用于输出文件名）
    score_column : str
        要提取的分数列名，默认为'score'
        
    Returns:
    --------
    pandas.DataFrame
        合并后的因子数据，行为日期，列为股票代码
    """
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(output_folder, "*.csv"))
    
    print(f"找到 {len(csv_files)} 个{market_name}数据文件")
    
    # 存储所有数据
    all_data = {}
    date_columns = set()
    
    for file_path in tqdm(csv_files, desc=f"处理{market_name}数据"):
        try:
            # 从文件名获取股票代码
            symbol = os.path.splitext(os.path.basename(file_path))[0]
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查必要的列是否存在
            if 'date' not in df.columns or score_column not in df.columns:
                print(f"警告: 文件 {symbol} 缺少必要列，跳过")
                continue
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 提取日期和分数列
            factor_data = df[['date', score_column]].copy()
            factor_data = factor_data.dropna(subset=[score_column])
            
            if len(factor_data) > 0:
                # 设置日期为索引
                factor_data.set_index('date', inplace=True)
                factor_data.rename(columns={score_column: symbol}, inplace=True)
                
                all_data[symbol] = factor_data
                date_columns.update(factor_data.index)
                
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错: {str(e)}")
            continue
    
    if not all_data:
        print(f"没有找到有效的{market_name}数据")
        return pd.DataFrame()
    
    # 合并所有数据
    print(f"合并{market_name}数据...")
    merged_df = pd.concat(all_data.values(), axis=1, sort=True)
    
    # 按日期排序
    merged_df = merged_df.sort_index()
    
    print(f"{market_name}数据合并完成:")
    print(f"  - 股票数量: {len(merged_df.columns)}")
    print(f"  - 日期范围: {merged_df.index.min()} 到 {merged_df.index.max()}")
    print(f"  - 数据形状: {merged_df.shape}")
    
    return merged_df

def main():
    """
    主函数：合并台股和美股因子数据
    """
    
    # 定义路径
    base_path = r"D:\大学文档\MAFM\IP\AtrDataCode"
    
    # 台股数据路径
    tw50_output_path = os.path.join(base_path, "tw50_processedOHLC", "output_data")
    
    # 美股数据路径  
    sp500_output_path = os.path.join(base_path, "sp500_processedOHLC_CRSP", "output_data")
    
    # 输出路径
    output_path = os.path.join(base_path, "merged_factors")
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 50)
    print("开始合并因子数据")
    print("=" * 50)
    
    # 处理台股数据
    print("\n1. 处理台股数据...")
    if os.path.exists(tw50_output_path):
        tw50_factors = merge_factors_by_market(tw50_output_path, "台股")
        if not tw50_factors.empty:
            tw50_output_file = os.path.join(output_path, "台股因子.csv")
            tw50_factors.to_csv(tw50_output_file, encoding='utf-8-sig')
            print(f"台股因子数据已保存到: {tw50_output_file}")
        else:
            print("台股数据为空，跳过")
    else:
        print(f"台股数据路径不存在: {tw50_output_path}")
    
    # 处理美股数据
    print("\n2. 处理美股数据...")
    if os.path.exists(sp500_output_path):
        sp500_factors = merge_factors_by_market(sp500_output_path, "美股")
        if not sp500_factors.empty:
            sp500_output_file = os.path.join(output_path, "美股因子.csv")
            sp500_factors.to_csv(sp500_output_file, encoding='utf-8-sig')
            print(f"美股因子数据已保存到: {sp500_output_file}")
        else:
            print("美股数据为空，跳过")
    else:
        print(f"美股数据路径不存在: {sp500_output_path}")
    
    print("\n" + "=" * 50)
    print("因子数据合并完成！")
    print("=" * 50)
    
    # 显示数据概览
    print("\n数据概览:")
    if os.path.exists(tw50_output_path):
        tw50_file = os.path.join(output_path, "台股因子.csv")
        if os.path.exists(tw50_file):
            tw50_df = pd.read_csv(tw50_file, index_col=0, parse_dates=True)
            print(f"台股因子: {tw50_df.shape[0]} 个交易日, {tw50_df.shape[1]} 只股票")
    
    if os.path.exists(sp500_output_path):
        sp500_file = os.path.join(output_path, "美股因子.csv")
        if os.path.exists(sp500_file):
            sp500_df = pd.read_csv(sp500_file, index_col=0, parse_dates=True)
            print(f"美股因子: {sp500_df.shape[0]} 个交易日, {sp500_df.shape[1]} 只股票")

if __name__ == "__main__":
    main()
