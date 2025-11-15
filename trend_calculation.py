import pandas as pd
import numpy as np
import os
import glob

def calculate_atr_factor(
    data_path: str,
    target: str = 'taiwan_stock',
    atr_period: int = 14
) -> pd.DataFrame:
    """
    计算ATR因子
    
    参数:
        data_path: 数据路径
        target: 'taiwan_stock' 或 'us_stock'
        atr_period: ATR计算周期，默认14
    
    返回:
        包含date, code, atr的DataFrame
    """
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"在路径 {data_path} 下未找到CSV文件")
    
    # 存储所有结果
    all_results = []
    
    for file in csv_files:
        try:
            # 读取CSV文件
            if target == 'us_stock':
                # 美股数据处理
                df = pd.read_csv(file, index_col=0, parse_dates=['DATE'], encoding='utf-8')
                df = df.rename(columns={
                    'DATE': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close'
                })
            else:  # taiwan_stock
                # 台股数据处理
                df = pd.read_csv(file,  parse_dates=['date'], encoding='utf-8')
                df = df.rename(columns={
                    'date': 'date',
                    'Adj_Open': 'open',
                    'Adj_High': 'high',
                    'Adj_Low': 'low',
                    'Adj_Close': 'close'
                })
            
            # 提取股票代码（从文件名）
            stock_code = os.path.basename(file).replace('.csv', '')
            
            # 计算True Range (TR)
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = np.abs(df['high'] - df['prev_close'])
            df['tr3'] = np.abs(df['low'] - df['prev_close'])
            df['tr'] = np.maximum(np.maximum(df['tr1'], df['tr2']), df['tr3'])
            
            # 计算ATR
            df['atr'] = df['tr'].rolling(window=atr_period, min_periods=1).mean()
            
            # 创建结果DataFrame
            result_df = pd.DataFrame({
                'date': df['date'],
                'code': stock_code,
                'atr': df['atr'].values
            })
            
            all_results.append(result_df)
            
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            continue
    
    if not all_results:
        raise ValueError("未能成功处理任何文件")
    
    # 合并所有结果
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 按日期和代码排序
    final_df = final_df.sort_values(['date', 'code']).reset_index(drop=True)
    
    return final_df


def process_all_stocks(target: str = 'taiwan_stock', atr_period: int = 14):
    """
    处理所有股票数据并计算ATR因子
    
    参数:
        target: 'taiwan_stock' 或 'us_stock'
        atr_period: ATR计算周期
    """
    
    # 设置数据路径
    if target == 'us_stock':
        data_path = r'D:\大学文档\MAFM\IP\AtrDataCode\sp500_processedOHLC_CRSP'
    elif target == 'taiwan_stock':
        data_path = r'D:\大学文档\MAFM\IP\AtrDataCode\tw50_processedOHLC'
    else:
        raise ValueError("target必须是'taiwan_stock'或'us_stock'")
    
    print(f"开始处理{target}数据...")
    print(f"数据路径: {data_path}")
    
    # 计算ATR因子
    atr_df = calculate_atr_factor(data_path, target=target, atr_period=atr_period)
    
    # 保存结果
    output_filename = f'{target}_atr_factor.csv'
    atr_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"ATR因子计算完成！")
    print(f"结果已保存至: {output_filename}")
    print(f"数据形状: {atr_df.shape}")
    print(f"\n前5行数据:")
    print(atr_df.head())
    print(f"\n后5行数据:")
    print(atr_df.tail())
    atr_df = atr_df.dropna(subset=['atr'])
    atr_df = pd.pivot_table(atr_df, index='date', columns='code', values='atr')
    return atr_df


if __name__ == "__main__":
    # 处理台股数据
    print("=" * 50)
    print("处理台股数据")
    print("=" * 50)
    taiwan_atr = process_all_stocks(target='taiwan_stock', atr_period=12)
    taiwan_atr.to_csv(r'D:\大学文档\MAFM\IP\AtrDataCode\merged_factors\taiwan_atr_factor.csv', index=True, encoding='utf-8')
    # 处理美股数据
    print("\n" + "=" * 50)
    print("处理美股数据")
    print("=" * 50)
    us_atr = process_all_stocks(target='us_stock', atr_period=12)
    us_atr.to_csv(r'D:\大学文档\MAFM\IP\AtrDataCode\merged_factors\us_atr_factor.csv', index=True,
                      encoding='utf-8')

    print("\n处理完成！")
