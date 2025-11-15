import glob
import os
import re

import pandas as pd

from alpha_operator import *
from 因子分析 import kline

def alpha45(close:pd.DataFrame,volume:pd.DataFrame) -> pd.DataFrame:
    # (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *  rank(correlation(sum(close, 5), sum(close, 20), 2))))
    term1 = -rank(ts_sum(delay(close,5),20)/20)
    term2 = ts_corr(close,volume,2)
    term3 = rank(ts_corr(ts_sum(close,5),ts_sum(close,20),2))
    res = term1*term2*term3
    shift = 20+5-1  # 确定下公式前多少行无意义（小于最小滚动窗口）
    return res[shift:]

def alpha55(close:pd.DataFrame,high:pd.DataFrame,low:pd.DataFrame, volume:pd.DataFrame) -> pd.DataFrame:
    # (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    term1 = (close - ts_min(low, 12))
    term2 = (ts_max(high, 12) - ts_min(low,12))
    term3 = rank(term1/term2)
    term4 = rank(volume)
    res = -ts_corr(term3,term4,6)
    shift = 12+6-1
    return res[shift:]

if __name__ == '__main__':
    file_list = glob.glob(os.path.dirname('./') + './sp500_processedOHLC_CRSP/*.csv')
    code_list = list(map(lambda x: re.findall(r'\\(\w+).csv',x)[0], file_list))
    us_kline = kline(code_list,start='20200101',target='us_stock')
    close_df = us_kline.get_data('收盘')
    volume_df = us_kline.get_data('成交量')
    high_df = us_kline.get_data('最高')
    low_df = us_kline.get_data('最低')
    alpha45_df = alpha45(close_df,volume_df)
    alpha55_df = alpha55(close_df,high_df,low_df,volume_df)
