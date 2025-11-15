import os
import glob
from cost_score import calculate_cost_score
from trend_calculation import calculate_mid_trend, calculate_short_trend
import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def load_sp500_data(folder_path):
    """
    读取SP500历史成分股数据
    
    Parameters:
    -----------
    folder_path : str
        包含CSV文件的文件夹路径
        
    Returns:
    --------
    dict
        键为标的名称（文件名），值为包含OHLC数据的DataFrame
    """
    data_dict = {}
    
    # 查找文件夹中的所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    print(f"找到 {len(csv_files)} 个数据文件")
    
    for file_path in csv_files:
        try:
            # 从文件名获取标的名称（去掉扩展名）
            symbol = os.path.splitext(os.path.basename(file_path))[0]
            
            # 读取CSV文件
            df = pd.read_csv(file_path, index_col=0)
            
            # 统一大小写
            df.columns = [col.lower() for col in df.columns]
            
            # 检查列是否存在
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                print(f"警告: 文件 {symbol} 缺少必要列，跳过")
                continue
            
            # 转换日期格式并排序！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df = df.sort_values('date', ascending=True)
            
            # # 设置日期为索引
            # df.set_index('date', inplace=True)
            
            # 确保数据类型正确
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['return'] = df['close'].pct_change(fill_method=None).shift(-1)
            df['price_diff'] = df['close'].diff().shift(-1)
            
            # # 删除包含NaN的行
            # # ????缺失值要直接删除吗
            # df = df.dropna(subset=required_columns[1:])
            
            if len(df) > 0:
                data_dict[symbol] = df
            else:
                print(f"警告: 文件 {symbol} 无有效数据，跳过")
                
        except Exception as e:
            print(f"错误: 读取文件 {file_path} 时出错: {str(e)}")
    
    return data_dict

def load_tw50_data(folder_path):
    """
    读取tw50历史成分股数据
    
    Parameters:
    -----------
    folder_path : str
        包含CSV文件的文件夹路径
        
    Returns:
    --------
    dict
        键为标的名称（文件名），值为包含OHLC数据的DataFrame
    """
    data_dict = {}
    
    # 查找文件夹中的所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    print(f"找到 {len(csv_files)} 个数据文件")
    
    for file_path in csv_files:
        # try:
        # 从文件名获取标的名称
        symbol = os.path.splitext(os.path.basename(file_path))[0]
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 统一大小写
        df.columns = [col.lower() for col in df.columns]
        df = df[['date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
        # 定义重命名映射字典
        rename_dict = {
            'adj_open': 'open',
            'adj_high': 'high', 
            'adj_low': 'low',
            'adj_close': 'close',
            'adj_volume': 'volume',
        }

        # 批量重命名
        df = df.rename(columns=rename_dict)

        # 检查必要的列是否存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"警告: 文件 {symbol} 缺少必要列，跳过")
            continue
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.sort_values('date', ascending=True)
 
        # # 设置日期为索引
        # df.set_index('date', inplace=True)

        # 确保数据类型正确
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['return'] = df['close'].pct_change(fill_method=None).shift(-1)
        df['price_diff'] = df['close'].diff().shift(-1)
        
        # # 删除包含NaN的行
        # # ????缺失值要直接删除吗
        # df = df.dropna(subset=required_columns[1:])
        
        if len(df) > 0:
            data_dict[symbol] = df
        else:
            print(f"警告: 文件 {symbol} 无有效数据，跳过")
            
        # except Exception as e:
        #     print(f"错误: 读取文件 {file_path} 时出错: {str(e)}")
    
    return data_dict

def calculate_score_batch(data_dict, n_workers, data_path):
    """
    批量计算score
    """
    results = {}
    kwargs = {'data_path': data_path}
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_symbol = {
            executor.submit(_calculate_single_symbol, symbol, df, **kwargs): symbol 
            for symbol, df in data_dict.items()
        }
        
        # 使用tqdm显示进度条
        with tqdm(total=len(future_to_symbol), desc="计算进度", unit="symbol") as pbar:
            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                # try:
                result_df = future.result()
                results[symbol] = result_df
                pbar.set_postfix(completed=symbol, refresh=False)
                pbar.update(1)
                # except Exception as e:
                #     print(f"\n错误: 计算 {symbol} 时出错: {str(e)}")
                #     pbar.update(1)
    
    return results

def _calculate_single_symbol(symbol, df, data_path):
    """cost, middle trend, short trend计算函数"""
    cost_params = {
    'bandLimitBase': 0.001,
    'bandLimitMult': 2.5, 
    'costAlloc': 10,
    'bb_period': 20,
    'bb_dev':2,
    'atr_period': 12 
    }
    mid_trend_params = {
        "date_col": "date",
        "close_col": "close",
        "short_ema_period": 5,
        "long_ema_period": 20,
        "trendMultMid": 2,
        "midTrendAlloc": 5
    }
    
    short_trend_params = {
        "date_col": "date",
        "open_col": "open",
        "high_col": "high",
        "low_col": "low",
        "close_col": "close",
        "atr_period": 12,
        "openMinBase": 0.001,
        "shortTrendAlloc": 10
    }
    ##################################### 分数计算函数入口 #####################################
    df_final = df[['date', 'open', 'high', 'low', 'close', "volume", "return", "price_diff"]].copy()
    # df = df.set_index('date')
    df_with_c = calculate_cost_score(df, **cost_params)
    df_with_mid = calculate_mid_trend(df, **mid_trend_params)
    df_with_short = calculate_short_trend(df, **short_trend_params)
    # df_final = df[['open', 'high', 'low', 'close', "volume", "return", "price_diff"]].join([
    #     df_with_c[['costScore']].rename(columns={'costScore': 'costScore'}),
    #     df_with_mid[['midTrendScore']].rename(columns={'midTrendScore': 'midTrendScore'}),
    #     df_with_short[['shortTrendScore']].rename(columns={'shortTrendScore': 'shortTrendScore'})
    # ])
    # df_with_c = df_with_c.reset_index(drop=False)
    # df_with_mid = df_with_mid.reset_index(drop=False)
    # df_with_short = df_with_short.reset_index(drop=False)
    df_final = df_final.merge(df_with_c[['date', 'costScore']], on='date', how='left')
    df_final = df_final.merge(df_with_mid[['date','midTrendScore']], on='date', how='left')
    df_final = df_final.merge(df_with_short[['date','shortTrendScore']], on='date', how='left')
    # df_final.set_index('date', inplace=True)
    # df_final = df[['open', 'high', 'low', 'close', "volume", "return", "price_diff"]]
    # df_final['costScore'] = df_with_c['costScore']
    # df_final['midTrendScore'] = df_with_mid['midTrendScore']
    # df_final['shortTrendScore'] = df_with_short['shortTrendScore']
    # if (len(df)!=len(df_with_c)) or (len(df)!=len(df_with_mid)) or (len(df)!=len(df_with_short)):
    #     print(len(df), len(df_with_mid), len(df_with_short)) 
    # 计算最终得分
    output_columns = ["date", "open", "high", "low", "close", "volume", "return", "price_diff", "midTrendScore", "shortTrendScore", "costScore"]
    df_output = df_final[output_columns].dropna(subset=["midTrendScore", "shortTrendScore", "costScore"])
    df_output = df_final
    df_output['score'] = (df_output['midTrendScore'] + df_output['shortTrendScore'] + df_output['costScore'])
    # 生成信号并计算pnl，保存数据到output文件夹下
    file_path = os.path.join(data_path, f"{symbol}.csv")
    trading_params = {
    "score_col": "score",
    "openThreshold": 0.5,
    "fullScore": 25,
    "save_path": file_path
    }
    analyze_results(df_output, **trading_params)

    return df_output

def analyze_results(df, score_col, openThreshold, fullScore, save_path):
    """
    生成pnl并存储数据
    """
    # 日期排序！！！！！！！！！！！！！！！！！！！！！！！
    # df_analysis = df.sort_index()
    df_analysis = df.sort_values(by='date').reset_index(drop=True)
    
    # 计算持仓信号
    df_analysis['signal'] = 0
    df_analysis.loc[df_analysis[score_col] > openThreshold * fullScore, 'signal'] = 1  # 开多头
    df_analysis.loc[df_analysis[score_col] < -openThreshold * fullScore, 'signal'] = -1  # 开空头
    # 生成持仓数据（简化为得到信号的下一个交易日建立仓位）
    df_analysis['pos'] = df_analysis['signal'].shift(1)
    # 生成pnl
    # df_analysis['pnl'] = df_analysis['pos'] * (df_analysis['close'].shift(-1) - df_analysis['close'])
    df_analysis['pnl'] = df_analysis['pos'] * (df_analysis['price_diff'])
    df_analysis['pnl'] = df_analysis['pnl'].fillna(0)
    df_analysis['cum_pnl'] = df_analysis['pnl'].cumsum()
    # 存储数据
    use_cols = ['date', 'open', 'high', 'low', 'close', "volume", "return", "price_diff",'costScore', "midTrendScore", "shortTrendScore", "score", "signal", "cum_pnl"]
    df_final = df_analysis[use_cols] 
    df_final.to_csv(save_path, index=True)  
    

def performance_test(folder_path, idx, data_path):
    import time
    start_load = time.time()

    # load data
    if idx == "sp500":
        data_dict = load_sp500_data(folder_path)
    elif idx == "tw50":
        data_dict = load_tw50_data(folder_path)

    end_load = time.time()
    load_time = end_load - start_load
    
    print(f"数据加载完成: {len(data_dict)} 个标的, 耗时 {load_time:.2f} 秒")
    
    if not data_dict:
        print("没有找到有效数据，退出")
        return
        
    # score calculation(parallel)
    start_calc = time.time()
    results = calculate_score_batch(data_dict, n_workers=4, data_path=data_path)
        
    end_calc = time.time()
    calc_time = end_calc - start_calc
    
    total_symbols = len(results)
    total_periods = sum(len(df) for df in results.values())
    
    print(f"处理标的数量: {total_symbols}")
    print(f"总数据周期: {total_periods}")
    print(f"数据加载时间: {load_time:.2f} 秒")
    print(f"指标计算时间: {calc_time:.2f} 秒")
    print(f"总时间: {load_time + calc_time:.2f} 秒")
    
    return results

if __name__ == "__main__":
    # 定义原始数据文件夹和标的指数名称
    folder_name = 'tw50_processedOHLC'
    data_folder = rf"D:/Python/HKUST_IP2/{folder_name}"  
    save_data_folder = rf"D:/Python/HKUST_IP2/{folder_name}/output_data"
    index = 'tw50'
    # 检查路径
    if not os.path.exists(save_data_folder):
        os.makedirs(save_data_folder, exist_ok=True)

    if not os.path.exists(data_folder):
        print(f"数据文件夹不存在: {data_folder}")
    else:
        # 主函数入口
        print("\n====== 开始计算 ======")
        all_results = performance_test(data_folder, index, save_data_folder)
        # print(all_results)
        # if all_results:
        #     trading_params = {
        #         "score_col": "score",
        #         "openThreshold": 0.5,
        #         "fullScore": 25
        #     }
            
        #     analyze_results(all_results, **trading_params)