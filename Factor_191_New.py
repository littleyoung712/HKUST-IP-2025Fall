import glob
import os
import re

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from 因子分析all import kline

# ==================== 增强的基础算子函数 ====================

def log(df: pd.DataFrame) -> pd.DataFrame:
    """自然对数，处理负值和零值"""
    return np.log(df.replace(0, np.nan)).fillna(0)

def power(df: pd.DataFrame, exp: Union[float, int, pd.DataFrame]) -> pd.DataFrame:
    """幂运算"""
    return df.pow(exp)

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """横截面排名（百分比排名）"""
    return df.rank(axis=1, pct=True)

def scale(df: pd.DataFrame) -> pd.DataFrame:
    """横截面标准化"""
    return df.div(df.abs().sum(axis=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)

def sign(df: pd.DataFrame) -> pd.DataFrame:
    """符号函数"""
    return np.sign(df)

def delay(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """span阶滞后项"""
    return df.shift(span)

def delta(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """时序差分"""
    return df.diff(span)

def ts_argmax(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """span内最大值对应的位置"""
    return df.rolling(span, min_periods=span).apply(lambda x: x.argmax() + 1 if len(x) == span else np.nan)

def ts_argmin(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """span内最小值对应的位置"""
    return df.rolling(span, min_periods=span).apply(lambda x: x.argmin() + 1 if len(x) == span else np.nan)

def ts_corr(df_1: pd.DataFrame, df_2: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序相关系数"""
    return df_1.rolling(span, min_periods=span).corr(df_2)

def ts_cov(df_1: pd.DataFrame, df_2: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序协方差"""
    return df_1.rolling(span, min_periods=span).cov(df_2)

def ts_max(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序最大值"""
    return df.rolling(span, min_periods=span).max()

def ts_mean(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序均值"""
    return df.rolling(span, min_periods=span).mean()

def ts_min(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序最小值"""
    return df.rolling(span, min_periods=span).min()

def ts_product(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序累乘"""
    return df.rolling(span, min_periods=span).apply(np.prod)

def ts_rank(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """最后一个值的时序排名（标准化）"""
    return df.rolling(window=span, min_periods=span).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x) == span else np.nan
    )

def ts_std(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序标准差"""
    return df.rolling(span, min_periods=span).std()

def ts_sum(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """时序求和"""
    return df.rolling(span, min_periods=span).sum()

def sma(df: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    """移动平均，SMA(x,n,m) = (m*x + (n-m)*SMA(x,n,m)_prev) / n"""
    alpha = m / n
    return df.ewm(alpha=alpha, adjust=False).mean()

def decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """线性衰减加权平均"""
    weights = np.arange(n, 0, -1)
    weights_sum = weights.sum()

    def weighted_average(x):
        if len(x) < n:
            return np.nan
        return np.sum(x * weights[:len(x)]) / weights_sum

    return df.rolling(n, min_periods=n).apply(weighted_average, raw=True)

def reg_beta(df: pd.DataFrame, x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """回归系数"""
    window = len(x)
    return df.rolling(window, min_periods=window).apply(
        lambda y: np.polyfit(x, y, deg=1)[0] if len(y) == window else np.nan
    )

def sum_if(df: pd.DataFrame, span: int, cond: pd.DataFrame) -> pd.DataFrame:
    """条件求和"""
    masked = df.where(cond, 0)
    return masked.rolling(span, min_periods=span).sum()

def max_df(df1, df2):
    """元素级最大值，支持DataFrame和标量的比较"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, (int, float)):
        return pd.DataFrame(np.maximum(df1.values, df2), 
                           index=df1.index, columns=df1.columns)
    elif isinstance(df1, (int, float)) and isinstance(df2, pd.DataFrame):
        return pd.DataFrame(np.maximum(df1, df2.values), 
                           index=df2.index, columns=df2.columns)
    elif isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return pd.DataFrame(np.maximum(df1.values, df2.values), 
                           index=df1.index, columns=df1.columns)
    else:
        raise ValueError("不支持的输入类型")

def min_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """元素级最小值"""
    return pd.DataFrame(np.minimum(df1.values, df2.values), 
                       index=df1.index, columns=df1.columns)

def abs_df(df):
    """DataFrame绝对值"""
    return pd.DataFrame(np.abs(df.values), index=df.index, columns=df.columns)

def sequence(n: int) -> np.ndarray:
    """生成1到n的序列"""
    return np.arange(1, n + 1)

def count(cond: pd.DataFrame, window: int) -> pd.DataFrame:
    """条件计数"""
    return cond.rolling(window, min_periods=window).sum()

def returns(df: pd.DataFrame) -> pd.DataFrame:
    """收益率"""
    return df.pct_change()

# ==================== 因子实现函数 ====================

def alpha100(volume, **kwargs):
    """
    Alpha #100: STD(VOLUME,20)
    """
    result = ts_std(volume, 20)
    shift = 20 - 1
    return result.iloc[shift:]

# ==================== Alpha 101-130 ====================

def alpha101(high, volume, vwap, close, **kwargs):
    """
    Alpha #101: ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(VOLUME), 11))) * -1)
    """
    part1 = rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15))
    weighted_high = (high * 0.1) + (vwap * 0.9)
    part2 = rank(ts_corr(rank(weighted_high), rank(volume), 11))
    
    condition = (part1 < part2)
    result = pd.DataFrame(np.where(condition, -1, 0), 
                         index=close.index, columns=close.columns)
    shift = max(30, 37, 15, 11) - 1
    return result.iloc[shift:]


def alpha102(volume, **kwargs):
    """
    Alpha #102: SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    """
    # 计算成交量变化
    volume_change = volume - delay(volume, 1)
    
    # 正成交量变化和绝对值变化
    positive_volume = max_df(volume_change, 0)
    abs_volume = abs_df(volume_change)
    
    # 计算加权移动平均
    sma_pos = sma(positive_volume, 6, 1)
    sma_abs = sma(abs_volume, 6, 1)
    
    # 计算结果
    result = sma_pos / sma_abs.replace(0, np.nan) * 100
    shift = 6 - 1
    return result.iloc[shift:]

def alpha103(low, **kwargs):
    """
    Alpha #103: ((20-LOWDAY(LOW,20))/20)*100
    """
    low_day = 20 - ts_argmin(low, 20)
    result = (low_day / 20) * 100
    shift = 20 - 1
    return result.iloc[shift:]

def alpha104(high, close, volume, **kwargs):
    """
    Alpha #104: (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
    """
    corr_delta = delta(ts_corr(high, volume, 5), 5)
    std_rank = rank(ts_std(close, 20))
    result = -1 * corr_delta * std_rank
    shift = max(5, 20) - 1
    return result.iloc[shift:]

def alpha105(open_price, volume, **kwargs):
    """
    Alpha #105: (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
    """
    result = -1 * ts_corr(rank(open_price), rank(volume), 10)
    shift = 10 - 1
    return result.iloc[shift:]

def alpha106(close, **kwargs):
    """
    Alpha #106: CLOSE-DELAY(CLOSE,20)
    """
    result = close - delay(close, 20)
    shift = 20
    return result.iloc[shift:]

def alpha107(open_price, high, low, close, **kwargs):
    """
    Alpha #107: (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
    """
    part1 = -1 * rank(open_price - delay(high, 1))
    part2 = rank(open_price - delay(close, 1))
    part3 = rank(open_price - delay(low, 1))
    result = part1 * part2 * part3
    shift = 1
    return result.iloc[shift:]

def alpha108(high, volume, vwap, **kwargs):
    """
    Alpha #108: ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)
    """
    part1 = rank(high - ts_min(high, 2))
    part2 = rank(ts_corr(vwap, ts_mean(volume, 120), 6))
    result = (power(part1, part2)) * -1
    shift = max(2, 120, 6) - 1
    return result.iloc[shift:]

def alpha109(high, low, **kwargs):
    """
    Alpha #109: SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    """
    price_range = high - low
    sma1 = sma(price_range, 10, 2)
    sma2 = sma(sma1, 10, 2)
    result = sma1 / sma2.replace(0, np.nan)
    shift = 10 - 1
    return result.iloc[shift:]

def alpha110(high, low, close, **kwargs):
    """
    Alpha #110: SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    """
    numerator = ts_sum(max_df(high - delay(close, 1), 0), 20)
    denominator = ts_sum(max_df(delay(close, 1) - low, 0), 20)
    result = numerator / denominator.replace(0, np.nan) * 100
    shift = 20 - 1
    return result.iloc[shift:]

def alpha111(high, low, close, volume, **kwargs):
    """
    Alpha #111: SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    """
    numerator = volume * ((close - low) - (high - close))
    denominator = (high - low).replace(0, np.nan)
    signal = numerator / denominator
    
    sma1 = sma(signal, 11, 2)
    sma2 = sma(signal, 4, 2)
    result = sma1 - sma2
    shift = max(11, 4) - 1
    return result.iloc[shift:]

def alpha112(close, **kwargs):
    """
    Alpha #112: (SUM((CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1):0),12) - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12) + SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    """
    price_diff = close - delay(close, 1)
    
    gains = pd.DataFrame(np.where(price_diff > 0, price_diff, 0), 
                       index=close.index, columns=close.columns)
    losses = pd.DataFrame(np.where(price_diff < 0, abs_df(price_diff), 0), 
                        index=close.index, columns=close.columns)
    
    sum_gains = ts_sum(gains, 12)
    sum_losses = ts_sum(losses, 12)
    
    result = (sum_gains - sum_losses) / (sum_gains + sum_losses).replace(0, np.nan) * 100
    shift = 12 - 1
    return result.iloc[shift:]

def alpha113(close, volume, **kwargs):
    """
    Alpha #113: (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))
    """
    part1 = rank(ts_sum(delay(close, 5), 20) / 20)
    part2 = ts_corr(close, volume, 2)
    part3 = rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))
    result = -1 * part1 * part2 * part3
    shift = max(5, 20, 2) - 1
    return result.iloc[shift:]

def alpha114(high, low, close, volume, vwap, **kwargs):
    """
    Alpha #114: ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    """
    range_normalized = (high - low) / (ts_sum(close, 5) / 5)
    numerator = rank(delay(range_normalized, 2)) * rank(rank(volume))
    denominator = range_normalized / (vwap - close).replace(0, np.nan)
    result = numerator / denominator.replace(0, np.nan)
    shift = max(5, 2) - 1
    return result.iloc[shift:]

def alpha115(high, low, close, volume, **kwargs):
    """
    Alpha #115: (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))
    """
    weighted_price = (high * 0.9) + (close * 0.1)
    part1 = rank(ts_corr(weighted_price, ts_mean(volume, 30), 10))
    
    mid_price = (high + low) / 2
    part2 = rank(ts_corr(ts_rank(mid_price, 4), ts_rank(volume, 10), 7))
    
    result = power(part1, part2)
    shift = max(30, 10, 4, 7) - 1
    return result.iloc[shift:]

def alpha116(close, **kwargs):
    """
    Alpha #116: REGBETA(CLOSE,SEQUENCE,20)
    """
    seq = sequence(20)
    result = reg_beta(close, seq)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha117(high, low, close, volume, returns, **kwargs):
    """
    Alpha #117: ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
    """
    part1 = ts_rank(volume, 32)
    part2 = 1 - ts_rank(((close + high) - low), 16)
    part3 = 1 - ts_rank(returns, 32)
    result = part1 * part2 * part3
    shift = max(32, 16) - 1
    return result.iloc[shift:]

def alpha118(open_price, high, low, **kwargs):
    """
    Alpha #118: SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    """
    numerator = ts_sum(high - open_price, 20)
    denominator = ts_sum(open_price - low, 20)
    result = numerator / denominator.replace(0, np.nan) * 100
    shift = 20 - 1
    return result.iloc[shift:]

def alpha119(open_price, volume, vwap, **kwargs):
    """
    Alpha #119: (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
    """
    part1_val = ts_corr(vwap, ts_sum(ts_mean(volume, 5), 26), 5)
    part1 = rank(decay_linear(part1_val, 7))
    
    corr_val = ts_corr(rank(open_price), rank(ts_mean(volume, 15)), 21)
    part2_val = ts_rank(ts_min(corr_val, 9), 7)
    part2 = rank(decay_linear(part2_val, 8))
    
    result = part1 - part2
    shift = max(5, 26, 5, 7, 15, 21, 9, 8) - 1
    return result.iloc[shift:]

def alpha120(vwap, close, **kwargs):
    """
    Alpha #120: (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
    """
    numerator = rank(vwap - close)
    denominator = rank(vwap + close)
    result = numerator / denominator.replace(0, np.nan)
    shift = 1
    return result.iloc[shift:]

def alpha121(vwap, volume, **kwargs):
    """
    Alpha #121: ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)
    逻辑：基于VWAP相对位置和量价关系时序排名的复杂策略
    """
    # 第一部分：VWAP相对于近期最低点的排名
    part1 = rank(vwap - ts_min(vwap, 12))
    
    # 第二部分：复杂的时序排名组合
    ts_rank_vwap = ts_rank(vwap, 20)  # VWAP的20日时序排名
    ts_rank_volume = ts_rank(ts_mean(volume, 60), 2)  # 60日平均成交量的2日时序排名
    corr_val = ts_corr(ts_rank_vwap, ts_rank_volume, 18)  # 18日相关系数
    part2 = ts_rank(corr_val, 3)  # 相关系数的3日时序排名
    
    # 组合信号：第一部分的结果以第二部分为指数，然后取负
    result = (power(part1, part2)) * -1
    
    # 最大窗口：max(12, 20, 60, 18, 3) = 60
    shift = 60 - 1
    return result.iloc[shift:]

def alpha122(close, **kwargs):
    """
    Alpha #122: (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    """
    log_close = log(close)
    sma1 = sma(log_close, 13, 2)
    sma2 = sma(sma1, 13, 2)
    sma3 = sma(sma2, 13, 2)
    
    numerator = sma3 - delay(sma3, 1)
    denominator = delay(sma3, 1)
    result = numerator / denominator.replace(0, np.nan)
    shift = max(13, 1) - 1
    return result.iloc[shift:]

def alpha123(high, low, volume, **kwargs):
    """
    Alpha #123: ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
    """
    mid_price = (high + low) / 2
    part1 = rank(ts_corr(ts_sum(mid_price, 20), ts_sum(ts_mean(volume, 60), 20), 9))
    part2 = rank(ts_corr(low, volume, 6))
    
    condition = (part1 < part2)
    result = pd.DataFrame(np.where(condition, -1, 0), 
                         index=high.index, columns=high.columns)
    shift = max(20, 60, 9, 6) - 1
    return result.iloc[shift:]

def alpha124(close, vwap, **kwargs):
    """
    Alpha #124: (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
    """
    numerator = close - vwap
    denominator = decay_linear(rank(ts_max(close, 30)), 2)
    result = numerator / denominator.replace(0, np.nan)
    shift = max(30, 2) - 1
    return result.iloc[shift:]

def alpha125(close, volume, vwap, **kwargs):
    """
    Alpha #125: (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))
    """
    part1 = rank(decay_linear(ts_corr(vwap, ts_mean(volume, 80), 17), 20))
    weighted_price = (close * 0.5) + (vwap * 0.5)
    part2 = rank(decay_linear(delta(weighted_price, 3), 16))
    result = part1 / part2.replace(0, np.nan)
    shift = max(80, 17, 20, 3, 16) - 1
    return result.iloc[shift:]

def alpha126(high, low, close, **kwargs):
    """
    Alpha #126: (CLOSE+HIGH+LOW)/3
    """
    result = (close + high + low) / 3
    shift = 1
    return result.iloc[shift:]

def alpha127(close, **kwargs):
    """
    Alpha #127: (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2),12)^(1/2)
    """
    numerator = 100 * (close - ts_max(close, 12))
    denominator = ts_max(close, 12)
    ratio = power(numerator / denominator.replace(0, np.nan), 2)
    mean_val = ts_mean(ratio, 12)
    result = power(mean_val, 0.5)
    shift = 12 - 1
    return result.iloc[shift:]

def alpha128(high, low, close, volume, **kwargs):
    """
    Alpha #128: 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
    """
    typical_price = (high + low + close) / 3
    cond_up = typical_price > delay(typical_price, 1)
    cond_down = typical_price < delay(typical_price, 1)
    
    volume_up = pd.DataFrame(np.where(cond_up, typical_price * volume, 0), 
                           index=high.index, columns=high.columns)
    volume_down = pd.DataFrame(np.where(cond_down, typical_price * volume, 0), 
                             index=high.index, columns=high.columns)
    
    sum_up = ts_sum(volume_up, 14)
    sum_down = ts_sum(volume_down, 14)
    
    ratio = sum_up / sum_down.replace(0, np.nan)
    result = 100 - (100 / (1 + ratio))
    shift = 14 - 1
    return result.iloc[shift:]

def alpha129(close, **kwargs):
    """
    Alpha #129: SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    """
    price_diff = close - delay(close, 1)
    losses = pd.DataFrame(np.where(price_diff < 0, abs_df(price_diff), 0), 
                        index=close.index, columns=close.columns)
    result = ts_sum(losses, 12)
    shift = 12 - 1
    return result.iloc[shift:]

def alpha130(high, low, volume, vwap, **kwargs):
    """
    Alpha #130: (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))
    """
    mid_price = (high + low) / 2
    part1 = rank(decay_linear(ts_corr(mid_price, ts_mean(volume, 40), 9), 10))
    part2 = rank(decay_linear(ts_corr(rank(vwap), rank(volume), 7), 3))
    result = part1 / part2.replace(0, np.nan)
    shift = max(40, 9, 10, 7, 3) - 1
    return result.iloc[shift:]

# ==================== Alpha 131-191 ====================

def alpha131(vwap, close, volume, **kwargs):
    """
    Alpha #131: (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
    """
    part1 = rank(delta(vwap, 1))
    part2 = ts_rank(ts_corr(close, ts_mean(volume, 50), 18), 18)
    result = power(part1, part2)
    shift = max(1, 50, 18) - 1
    return result.iloc[shift:]

def alpha132(volume, **kwargs):
    """
    Alpha #132: MEAN(AMOUNT,20)
    """
    result = ts_mean(volume, 20)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha133(high, low, **kwargs):
    """
    Alpha #133: ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    """
    high_day = 20 - ts_argmax(high, 20)
    low_day = 20 - ts_argmin(low, 20)
    result = (high_day / 20 * 100) - (low_day / 20 * 100)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha134(close, volume, **kwargs):
    """
    Alpha #134: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    """
    price_change = (close - delay(close, 12)) / delay(close, 12)
    result = price_change * volume
    shift = 12
    return result.iloc[shift:]

def alpha135(close, **kwargs):
    """
    Alpha #135: SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    """
    ratio = close / delay(close, 20)
    delayed_ratio = delay(ratio, 1)
    result = sma(delayed_ratio, 20, 1)
    shift = max(20, 1) - 1
    return result.iloc[shift:]

def alpha136(open_price, volume, returns, **kwargs):
    """
    Alpha #136: ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    """
    part1 = -1 * rank(delta(returns, 3))
    part2 = ts_corr(open_price, volume, 10)
    result = part1 * part2
    shift = max(3, 10) - 1
    return result.iloc[shift:]

def alpha137(open_price, high, low, close, **kwargs):
    """
    Alpha #137: 16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1)) / 
                (条件分母) * MAX(ABS(HIGH-DELAY(CLOSE,1)), ABS(LOW-DELAY(CLOSE,1)))
    逻辑：基于价格变动和波动率调整的复杂动量指标
    """
    # 分子部分：16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))
    # 简化后等于：16*(1.5*CLOSE - 0.5*OPEN - DELAY(OPEN,1))
    numerator = 16 * (1.5 * close - 0.5 * open_price - delay(open_price, 1))
    
    # 分母部分的条件判断
    A = abs_df(high - delay(close, 1))      # |HIGH - DELAY(CLOSE,1)|
    B = abs_df(low - delay(close, 1))       # |LOW - DELAY(CLOSE,1)|
    C = abs_df(high - delay(low, 1))        # |HIGH - DELAY(LOW,1)|
    D = abs_df(delay(close, 1) - delay(open_price, 1))  # |DELAY(CLOSE,1) - DELAY(OPEN,1)|
    
    # 条件1: A > B 且 A > C
    cond1 = (A > B) & (A > C)
    # 条件2: B > C 且 B > A  
    cond2 = (B > C) & (B > A)
    # 条件3: 其他情况
    cond3 = ~cond1 & ~cond2
    
    # 根据条件计算分母
    denominator = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    
    # 条件1: A + B/2 + D/4
    denominator[cond1] = A[cond1] + B[cond1]/2 + D[cond1]/4
    
    # 条件2: B + A/2 + D/4  
    denominator[cond2] = B[cond2] + A[cond2]/2 + D[cond2]/4
    
    # 条件3: C + D/4
    denominator[cond3] = C[cond3] + D[cond3]/4
    
    # 乘以最大值部分：MAX(ABS(HIGH-DELAY(CLOSE,1)), ABS(LOW-DELAY(CLOSE,1)))
    max_part = max_df(A, B)
    
    # 最终结果
    result = numerator / denominator.replace(0, np.nan) * max_part
    
    # 窗口：需要delay(1)，所以shift=1
    shift = 1
    return result.iloc[shift:]

def alpha138(low, volume, vwap, **kwargs):
    """
    Alpha #138: ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)
    """
    weighted_price = (low * 0.7) + (vwap * 0.3)
    part1 = rank(decay_linear(delta(weighted_price, 3), 20))
    
    corr_val = ts_corr(ts_rank(low, 8), ts_rank(ts_mean(volume, 60), 17), 5)
    part2 = ts_rank(decay_linear(ts_rank(corr_val, 19), 16), 7)
    
    result = (part1 - part2) * -1
    shift = max(3, 20, 8, 60, 17, 5, 19, 16, 7) - 1
    return result.iloc[shift:]

def alpha139(open_price, volume, **kwargs):
    """
    Alpha #139: (-1 * CORR(OPEN, VOLUME, 10))
    """
    result = -1 * ts_corr(open_price, volume, 10)
    shift = 10 - 1
    return result.iloc[shift:]

def alpha140(open_price, high, low, close, volume, **kwargs):
    """
    Alpha #140: MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))
    """
    part1_val = (rank(open_price) + rank(low)) - (rank(high) + rank(close))
    part1 = rank(decay_linear(part1_val, 8))
    
    corr_val = ts_corr(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 20), 8)
    part2 = ts_rank(decay_linear(corr_val, 7), 3)
    
    result = min_df(part1, part2)
    shift = max(8, 60, 20, 7, 3) - 1
    return result.iloc[shift:]

def alpha141(high, volume, **kwargs):
    """
    Alpha #141: (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
    """
    result = rank(ts_corr(rank(high), rank(ts_mean(volume, 15)), 9)) * -1
    shift = max(15, 9) - 1
    return result.iloc[shift:]

def alpha142(close, volume, **kwargs):
    """
    Alpha #142: (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))
    """
    part1 = -1 * rank(ts_rank(close, 10))
    part2 = rank(delta(delta(close, 1), 1))
    part3 = rank(ts_rank(volume / ts_mean(volume, 20), 5))
    result = part1 * part2 * part3
    shift = max(10, 20, 5) - 1
    return result.iloc[shift:]

def alpha143(close, **kwargs):
    """
    Alpha #143: CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    逻辑：递归的价格动量指标，当价格上涨时计算收益率并累乘，否则保持原值
    """
    # 递归实现这个因子比较复杂，这里使用简化版本：
    # 计算价格是否上涨的条件
    cond = close > delay(close, 1)
    
    # 当价格上涨时：计算收益率并乘以自身（这里用累计乘积近似）
    # 当价格下跌时：保持原值（这里用1表示）
    price_change = (close - delay(close, 1)) / delay(close, 1)
    
    # 初始化结果
    result = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    
    # 递归计算（使用循环实现）
    for i in range(1, len(close)):
        for col in close.columns:
            if cond.iloc[i][col]:
                # 价格上涨：乘以(1 + 收益率)
                result.iloc[i][col] = result.iloc[i-1][col] * (1 + price_change.iloc[i][col])
            else:
                # 价格下跌：保持原值
                result.iloc[i][col] = result.iloc[i-1][col]
    shift = 1
    return result.iloc[shift:]

def alpha144(close, volume, **kwargs):
    """
    Alpha #144: SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    """
    ret_abs = abs_df(close / delay(close, 1) - 1)
    normalized = ret_abs /volume.replace(0, np.nan)
    
    cond = (close < delay(close, 1))
    numerator = sum_if(normalized, 20, cond)
    denominator = count(cond, 20)
    result = numerator / denominator.replace(0, np.nan)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha145(volume, **kwargs):
    """
    Alpha #145: (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    """
    numerator = ts_mean(volume, 9) - ts_mean(volume, 26)
    denominator = ts_mean(volume, 12)
    result = numerator / denominator.replace(0, np.nan) * 100
    shift = max(9, 26, 12) - 1
    return result.iloc[shift:]

def alpha146(close, **kwargs):
    """
    Alpha #146: 
    MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20) * 
    ((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)) / 
    SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,61,2)
    
    逻辑：基于收益率与其SMA偏离度的标准化动量指标
    """
    # 计算日收益率
    returns = (close - delay(close, 1)) / delay(close, 1)
    
    # 计算收益率的SMA(61,2)
    sma_returns = sma(returns, 61, 2)
    
    # 计算收益率与SMA的偏离
    deviation = returns - sma_returns
    
    # 第一部分：偏离的20日均值
    part1 = ts_mean(deviation, 20)
    
    # 第二部分：当前的偏离值
    part2 = deviation
    
    # 第三部分：偏离平方的SMA(61,2)
    # 注意：公式中这部分有重复计算，应该是 deviation 的平方
    squared_deviation = power(deviation, 2)
    part3 = sma(squared_deviation, 61, 2)
    
    # 组合计算： (均值偏离 * 当前偏离) / 偏离方差
    result = (part1 * part2) / part3.replace(0, np.nan)
    
    # 最大窗口：max(61, 20) = 61
    shift = 61 - 1
    return result.iloc[shift:]

def alpha147(close, **kwargs):
    """
    Alpha #147: REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    """
    mean_close = ts_mean(close, 12)
    seq = sequence(12)
    result = reg_beta(mean_close, seq)
    shift = 12 - 1
    return result.iloc[shift:]

def alpha148(open_price, volume, **kwargs):
    """
    Alpha #148: ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
    """
    part1 = rank(ts_corr(open_price, ts_sum(ts_mean(volume, 60), 9), 6))
    part2 = rank(open_price - ts_min(open_price, 14))
    
    condition = (part1 < part2)
    result = pd.DataFrame(np.where(condition, -1, 0), 
                         index=open_price.index, columns=open_price.columns)
    shift = max(60, 9, 6, 14) - 1
    return result.iloc[shift:]

def alpha149(close, benchmark_close, **kwargs):
    """
    Alpha #149: 
    REGBETA(
        FILTER(CLOSE/DELAY(CLOSE,1)-1, BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),
        FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1, BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),
        252
    )
    逻辑：在基准下跌的日子里，计算个股收益率对基准收益率的回归系数（252天）
    """
    # 计算个股和基准的日收益率
    stock_ret = close / delay(close, 1) - 1
    bench_ret = benchmark_close / delay(benchmark_close, 1) - 1
    
    # 筛选条件：基准下跌的日子
    bench_down_condition = benchmark_close < delay(benchmark_close, 1)
    
    # 过滤数据：只在基准下跌的日子里保留收益率数据，其他日子设为NaN
    filtered_stock_ret = stock_ret.where(bench_down_condition, np.nan)
    filtered_bench_ret = bench_ret.where(bench_down_condition, np.nan)
    
    # 计算回归系数（252天窗口）
    # 由于reg_beta函数需要完整的序列，我们需要滚动计算
    def calc_beta_window(stock_series, bench_series):
        """计算252天窗口的回归系数"""
        if len(stock_series) < 252:
            return np.nan
        
        # 移除NaN值
        valid_mask = ~np.isnan(stock_series) & ~np.isnan(bench_series)
        valid_stock = stock_series[valid_mask]
        valid_bench = bench_series[valid_mask]
        
        if len(valid_stock) < 50:  # 最少需要一定数量的有效数据
            return np.nan
            
        # 计算回归系数
        try:
            beta = np.polyfit(valid_bench, valid_stock, 1)[0]
            return beta
        except:
            return np.nan
    
    # 对每只股票计算beta
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    
    for col in close.columns:
        # 获取该股票的过滤后收益率序列
        stock_series = filtered_stock_ret[col]
        bench_series = filtered_bench_ret.iloc[:, 0]  # 基准只有一列
        
        # 滚动计算252天beta
        for i in range(251, len(stock_series)):
            window_stock = stock_series.iloc[i-251:i+1]
            window_bench = bench_series.iloc[i-251:i+1]
            result.iloc[i][col] = calc_beta_window(window_stock, window_bench)
    shift = 252 - 1
    return result.iloc[shift:]

def alpha150(high, low, close, volume, **kwargs):
    """
    Alpha #150: (CLOSE+HIGH+LOW)/3*VOLUME
    """
    typical_price = (close + high + low) / 3
    result = typical_price * volume
    shift = 1
    return result.iloc[shift:]

def alpha151(close, **kwargs):
    """
    Alpha #151: SMA(CLOSE-DELAY(CLOSE,20),20,1)
    """
    price_diff = close - delay(close, 20)
    result = sma(price_diff, 20, 1)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha152(close, **kwargs):
    """
    Alpha #152: 
    SMA(
        MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12) - 
        MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),
        9,1
    )
    逻辑：基于9日收益率的多重平滑移动平均差异
    """
    # 第一步：计算9日收益率 CLOSE/DELAY(CLOSE,9)
    returns_9d = close / delay(close, 9)
    
    # 第二步：DELAY(...,1) 滞后1天
    delayed_returns = delay(returns_9d, 1)
    
    # 第三步：SMA(...,9,1) 对滞后后的9日收益率进行SMA平滑
    sma_returns = sma(delayed_returns, 9, 1)
    
    # 第四步：DELAY(...,1) 再次滞后1天
    delayed_sma = delay(sma_returns, 1)
    
    # 第五步：计算12日和26日的均值
    mean_12 = ts_mean(delayed_sma, 12)   # 12日均值
    mean_26 = ts_mean(delayed_sma, 26)   # 26日均值
    
    # 第六步：计算差异
    diff = mean_12 - mean_26
    
    # 第七步：对差异进行SMA(9,1)平滑
    result = sma(diff, 9, 1)
    
    # 计算最大窗口：max(9, 1, 9, 1, 12, 26, 9) = 26 + 各个延迟 = 约35天
    shift = 35 - 1
    return result.iloc[shift:]

def alpha153(close, **kwargs):
    """
    Alpha #153: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    """
    result = (ts_mean(close, 3) + ts_mean(close, 6) + 
             ts_mean(close, 12) + ts_mean(close, 24)) / 4
    shift = 24 - 1
    return result.iloc[shift:]

def alpha154(vwap, volume, **kwargs):
    """
    Alpha #154: (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))
    逻辑：判断VWAP相对位置与量价相关性的条件关系
    """
    part1 = vwap - ts_min(vwap, 16)  # VWAP相对于16日最低点的位置
    part2 = ts_corr(vwap, ts_mean(volume, 180), 18)  # VWAP与180日均量量的18日相关性
    
    # 条件判断：如果VWAP相对位置小于量价相关性，则为1，否则为0
    condition = part1 < part2
    result = pd.DataFrame(np.where(condition, 1, 0), 
                         index=vwap.index, columns=vwap.columns)
    
    shift = max(16, 180, 18) - 1
    return result.iloc[shift:]

def alpha155(volume, **kwargs):
    """
    Alpha #155: SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    """
    sma13 = sma(volume, 13, 2)
    sma27 = sma(volume, 27, 2)
    diff = sma13 - sma27
    sma_diff = sma(diff, 10, 2)
    result = sma13 - sma27 - sma_diff
    shift = max(13, 27, 10) - 1
    return result.iloc[shift:]

def alpha156(vwap, open_price, low, **kwargs):
    """
    Alpha #156: (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)
    """
    part1 = rank(decay_linear(delta(vwap, 5), 3))
    
    weighted_price = (open_price * 0.15) + (low * 0.85)
    ratio = delta(weighted_price, 2) / weighted_price.replace(0, np.nan)
    part2 = rank(decay_linear(ratio * -1, 3))
    
    result = max_df(part1, part2) * -1
    shift = max(5, 3, 2) - 1
    return result.iloc[shift:]

def alpha157(close, returns, **kwargs):
    """
    Alpha #157: 
    (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + 
     TSRANK(DELAY((-1 * RET), 6), 5))
    逻辑：复杂排名组合与收益率时序排名的加总
    """
    # 第一部分：复杂的多层排名嵌套
    inner_rank = rank(rank(-1 * rank(delta(close - 1, 5))))
    ts_min_rank = ts_min(inner_rank, 2)
    sum_rank = ts_sum(ts_min_rank, 1)
    log_rank = log(sum_rank.replace(0, np.nan))
    rank_rank = rank(rank(log_rank))
    prod_rank = ts_product(rank_rank, 1)
    min_prod = ts_min(prod_rank, 5)
    
    # 第二部分：负收益率的时序排名
    neg_returns = -1 * returns
    delayed_neg_returns = delay(neg_returns, 6)
    ts_rank_returns = ts_rank(delayed_neg_returns, 5)
    
    # 组合结果
    result = min_prod + ts_rank_returns
    
    shift = max(5, 2, 6, 5) - 1
    return result.iloc[shift:]

def alpha158(high, low, close, **kwargs):
    """
    Alpha #158: ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    """
    sma_close = sma(close, 15, 2)
    high_diff = high - sma_close
    low_diff = low - sma_close
    result = (high_diff - low_diff) / close.replace(0, np.nan)
    shift = 15 - 1
    return result.iloc[shift:]

def alpha159(high, low, close, **kwargs):
    """
    Alpha #159:
    ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24 +
     (CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24 +
     (CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24) * 
     100/(6*12+6*24+12*24)
    逻辑：多时间窗口的价格位置效率指标
    """
    delay_close = delay(close, 1)
    
    # 计算公共部分
    min_low_close = min_df(low, delay_close)  # MIN(LOW, DELAY(CLOSE,1))
    max_high_close = max_df(high, delay_close)  # MAX(HIGH, DELAY(CLOSE,1))
    price_range = max_high_close - min_low_close  # 价格区间
    
    # 6日窗口部分
    sum_min_6 = ts_sum(min_low_close, 6)
    sum_range_6 = ts_sum(price_range, 6)
    part_6 = (close - sum_min_6) / sum_range_6.replace(0, np.nan) * 12 * 24
    
    # 12日窗口部分
    sum_min_12 = ts_sum(min_low_close, 12)
    sum_range_12 = ts_sum(price_range, 12)
    part_12 = (close - sum_min_12) / sum_range_12.replace(0, np.nan) * 6 * 24
    
    # 24日窗口部分
    sum_min_24 = ts_sum(min_low_close, 24)
    sum_range_24 = ts_sum(price_range, 24)
    part_24 = (close - sum_min_24) / sum_range_24.replace(0, np.nan) * 6 * 24
    
    # 组合结果
    numerator = part_6 + part_12 + part_24
    denominator = (6*12 + 6*24 + 12*24)  # 432
    result = numerator * 100 / denominator
    
    shift = 24 - 1
    return result.iloc[shift:]

def alpha160(close, **kwargs):
    """
    Alpha #160: SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    """
    cond = (close <= delay(close, 1))
    signal = pd.DataFrame(np.where(cond, ts_std(close, 20), 0), 
                         index=close.index, columns=close.columns)
    result = sma(signal, 20, 1)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha161(high, low, close, **kwargs):
    """
    Alpha #161: MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    """
    range1 = high - low
    range2 = abs_df(delay(close, 1) - high)
    range3 = abs_df(delay(close, 1) - low)
    max_range = max_df(max_df(range1, range2), range3)
    result = ts_mean(max_range, 12)
    shift = max(1, 12) - 1
    return result.iloc[shift:]

def alpha162(close, **kwargs):
    """
    Alpha #162: RSI类型的标准化指标
    (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100 - 
     MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)) / 
    (MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12) - 
     MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    """
    # 计算RSI类型的指标
    price_diff = close - delay(close, 1)
    gains = max_df(price_diff, 0)
    abs_diff = abs_df(price_diff)
    
    # 计算平滑后的增益和绝对变化
    sma_gains = sma(gains, 12, 1)
    sma_abs = sma(abs_diff, 12, 1)
    
    # 计算RSI值
    rsi = sma_gains / sma_abs.replace(0, np.nan) * 100
    
    # 标准化：将RSI缩放到0-1范围
    min_rsi = ts_min(rsi, 12)
    max_rsi = ts_max(rsi, 12)
    result = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    
    shift = 12 - 1
    return result.iloc[shift:]

def alpha163(close, volume, vwap, returns, **kwargs):
    """
    Alpha #163: RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    """
    # 注意：这里需要high数据，但参数中没有，使用close近似
    part = ((-1 * returns) * ts_mean(volume, 20)) * vwap * (close - delay(close, 1))
    result = rank(part)
    shift = max(20, 1) - 1
    return result.iloc[shift:]

def alpha164(close, high, low, **kwargs):
    """
    Alpha #164: 
    SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) - 
         MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12)) / 
        (HIGH-LOW)*100, 13, 2)
    逻辑：基于价格变化的倒数与价格区间的标准化指标
    """
    cond = close > delay(close, 1)
    price_diff = close - delay(close, 1)
    
    # 计算条件值：上涨时为1/涨幅，否则为1
    signal = pd.DataFrame(np.where(cond, 1 / price_diff.replace(0, np.nan), 1), 
                         index=close.index, columns=close.columns)
    
    # 减去12日最小值
    signal_min = ts_min(signal, 12)
    signal_diff = signal - signal_min
    
    # 除以价格区间并乘以100
    price_range = high - low
    normalized = signal_diff / price_range.replace(0, np.nan) * 100
    
    # 最终平滑
    result = sma(normalized, 13, 2)
    
    shift = max(12, 13) - 1
    return result.iloc[shift:]

def alpha165(close, **kwargs):
    """
    Alpha #165: 
    MAX(SUMAC(CLOSE-MEAN(CLOSE,48))) - MIN(SUMAC(CLOSE-MEAN(CLOSE,48))) / STD(CLOSE,48)
    逻辑：累积偏离的极差与波动率比率
    """
    mean_close = ts_mean(close, 48)
    deviation = close - mean_close
    
    # 计算累积和
    cum_sum = ts_sum(deviation, 48)
    
    # 计算最大值和最小值
    max_cum = ts_max(cum_sum, 48)
    min_cum = ts_min(cum_sum, 48)
    
    # 计算标准差
    std_close = ts_std(close, 48)
    
    result = max_cum - min_cum / std_close.replace(0, np.nan)
    
    shift = 48 - 1
    return result.iloc[shift:]

def alpha166(close, **kwargs):
    """
    Alpha #166: 
    -20*(20-1)^1.5 * SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20) / 
    ((20-1)*(20-2) * (SUM((CLOSE/DELAY(CLOSE,1))^2,20))^1.5)
    
    逻辑：收益率的偏度计算（标准化三阶矩）
    """
    # 计算日收益率
    returns = close / delay(close, 1) - 1
    
    # 计算20日平均收益率
    mean_returns = ts_mean(returns, 20)
    
    # 分子部分：SUM(收益率 - 平均收益率, 20)
    deviation_sum = ts_sum(returns - mean_returns, 20)
    
    # 分子系数：-20*(19)^1.5
    numerator_coef = -20 * (19 ** 1.5)
    numerator = numerator_coef * deviation_sum
    
    # 分母第一部分：(20-1)*(20-2) = 19*18
    denominator_coef = 19 * 18
    
    # 分母第二部分：SUM(收益率^2, 20)^1.5
    squared_returns_sum = ts_sum(returns ** 2, 20)
    denominator_power = power(squared_returns_sum, 1.5)
    
    # 完整分母
    denominator = denominator_coef * denominator_power
    
    # 最终结果
    result = numerator / denominator.replace(0, np.nan)
    
    shift = 20 - 1
    return result.iloc[shift:]

def alpha167(close, **kwargs):
    """
    Alpha #167: SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    """
    price_diff = close - delay(close, 1)
    gains = pd.DataFrame(np.where(price_diff > 0, price_diff, 0), 
                       index=close.index, columns=close.columns)
    result = ts_sum(gains, 12)
    shift = 12 - 1
    return result.iloc[shift:]

def alpha168(volume, **kwargs):
    """
    Alpha #168: (-1*VOLUME/MEAN(VOLUME,20))
    """
    result = -1 * volume / ts_mean(volume, 20)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha169(close, **kwargs):
    """
    Alpha #169: 移动平均收敛
    """
    ma_fast = ts_mean(close, 5)
    ma_slow = ts_mean(close, 10)
    result = (ma_fast - ma_slow) / ma_slow
    shift = 10 - 1
    return result.iloc[shift:]

def alpha170(close, volume, high, vwap, **kwargs):
    """
    Alpha #170:
    ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * 
      ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - 
     RANK((VWAP - DELAY(VWAP, 5))))
    逻辑：价格倒数排名、成交量、高价排名和VWAP变化的综合指标
    """
    # 第一部分：价格倒数排名与成交量标准化
    part1 = (rank(1 / close) * volume) / ts_mean(volume, 20)
    
    # 第二部分：高价排名比率
    part2 = (high * rank(high - close)) / (ts_sum(high, 5) / 5)
    
    # 第三部分：VWAP变化排名
    part3 = rank(vwap - delay(vwap, 5))
    
    result = part1 * part2 - part3
    
    shift = max(20, 5, 5) - 1
    return result.iloc[shift:]

def alpha171(open_price, high, low, close, **kwargs):
    """
    Alpha #171: 开盘缺口反转
    """
    gap = open_price - delay(close, 1)
    result = -1 * gap / delay(close, 1)
    shift = 1
    return result.iloc[shift:]

def alpha172(high, low, close, volume, **kwargs):
    """
    Alpha #172: ATR类型的波动率指标
    """
    tr1 = high - low
    tr2 = abs_df(high - delay(close, 1))
    tr3 = abs_df(low - delay(close, 1))
    true_range = max_df(max_df(tr1, tr2), tr3)
    result = ts_mean(true_range, 14)
    shift = max(1, 14) - 1
    return result.iloc[shift:]

def alpha173(close, **kwargs):
    """
    Alpha #173: 三重移动平均
    """
    ma1 = ts_mean(close, 5)
    ma2 = ts_mean(close, 10)
    ma3 = ts_mean(close, 20)
    result = (ma1 + ma2 + ma3) / 3
    shift = 20 - 1
    return result.iloc[shift:]

def alpha174(close, **kwargs):
    """
    Alpha #174: 条件波动率
    """
    cond = (close > delay(close, 1))
    signal = pd.DataFrame(np.where(cond, ts_std(close, 20), 0), 
                         index=close.index, columns=close.columns)
    result = sma(signal, 20, 1)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha175(high, low, close, **kwargs):
    """
    Alpha #175: 平均真实波幅
    """
    tr1 = high - low
    tr2 = abs_df(high - delay(close, 1))
    tr3 = abs_df(low - delay(close, 1))
    true_range = max_df(max_df(tr1, tr2), tr3)
    result = ts_mean(true_range, 6)
    shift = max(1, 6) - 1
    return result.iloc[shift:]

def alpha176(high, low, close, volume, **kwargs):
    """
    Alpha #176: 价格量能结合
    """
    price_position = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))
    result = ts_corr(rank(price_position), rank(volume), 6)
    shift = max(12, 6) - 1
    return result.iloc[shift:]

def alpha177(high, **kwargs):
    """
    Alpha #177: 高点突破
    """
    result = ((20 - ts_argmax(high, 20)) / 20) * 100
    shift = 20 - 1
    return result.iloc[shift:]

def alpha178(close, volume, **kwargs):
    """
    Alpha #178: 价格变化率
    """
    result = (close - delay(close, 1)) / delay(close, 1) * volume
    shift = 1
    return result.iloc[shift:]

def alpha179(low, volume, vwap, **kwargs):
    """
    Alpha #179: 低点量价关系
    """
    result = rank(ts_corr(vwap, volume, 4)) * rank(ts_corr(rank(low), rank(ts_mean(volume, 50)), 12))
    shift = max(4, 50, 12) - 1
    return result.iloc[shift:]

def alpha180(close, volume, adv20, **kwargs):
    """
    Alpha #180: 条件量价动量
    """
    condition = (adv20 < volume)
    delta_close = delta(close, 7)
    ts_rank_val = ts_rank(abs_df(delta_close), 60)
    signal = (-1 * ts_rank_val) * sign(delta_close)
    result = pd.DataFrame(np.where(condition, signal, -1 * volume), 
                         index=close.index, columns=close.columns)
    shift = max(7, 60) - 1
    return result.iloc[shift:]

def alpha181(close, benchmark_close, **kwargs):
    """
    Alpha #181: 相对动量
    """
    stock_ret = close / delay(close, 1) - 1
    bench_ret = benchmark_close / delay(benchmark_close, 1) - 1
    result = ts_sum((stock_ret - bench_ret) ** 2, 20) / 20
    shift = max(1, 20) - 1
    return result.iloc[shift:]

def alpha182(close, open_price, benchmark_close, benchmark_open, **kwargs):
    """
    Alpha #182: 同步性指标
    """
    stock_sync = (close > open_price) == (benchmark_close > benchmark_open)
    result = count(stock_sync, 20) / 20
    shift = 20 - 1
    return result.iloc[shift:]

def alpha183(close, **kwargs):
    """
    Alpha #183: 累积偏离
    """
    mean_close = ts_mean(close, 24)
    cum_deviation = ts_sum(close - mean_close, 24)
    result = cum_deviation / (ts_std(close, 24) * np.sqrt(24)).replace(0, np.nan)
    shift = 24 - 1
    return result.iloc[shift:]

def alpha184(open_price, close, **kwargs):
    """
    Alpha #184: 开盘收盘关系
    """
    result = rank(ts_corr(delay(open_price - close, 1), close, 200)) + rank(open_price - close)
    shift = max(1, 200) - 1
    return result.iloc[shift:]

def alpha185(open_price, close, **kwargs):
    """
    Alpha #185: 开盘缺口
    """
    result = rank((-1 * ((1 - (open_price / close)) ** 2)))
    shift = 1
    return result.iloc[shift:]

def alpha186(high, low, close, volume, **kwargs):
    """
    Alpha #186: 威廉指标类型
    """
    highest_high = ts_max(high, 14)
    lowest_low = ts_min(low, 14)
    result = (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan) * 100
    shift = 14 - 1
    return result.iloc[shift:]

def alpha187(open_price, high, **kwargs):
    """
    Alpha #187: 开盘突破
    """
    cond = (open_price <= delay(open_price, 1))
    signal = pd.DataFrame(np.where(cond, 0, 
                         max_df(high - open_price, open_price - delay(open_price, 1))), 
                         index=open_price.index, columns=open_price.columns)
    result = ts_sum(signal, 20)
    shift = 20 - 1
    return result.iloc[shift:]

def alpha188(high, low, **kwargs):
    """
    Alpha #188: 波动率变化
    """
    range_ = high - low
    sma_range = sma(range_, 11, 2)
    result = ((range_ - sma_range) / sma_range.replace(0, np.nan)) * 100
    shift = 11 - 1
    return result.iloc[shift:]

def alpha189(close, **kwargs):
    """
    Alpha #189: 平均绝对偏差
    """
    result = ts_mean(abs_df(close - ts_mean(close, 6)), 6)
    shift = 6 - 1
    return result.iloc[shift:]

def alpha190(close, **kwargs):
    """
    Alpha #190: 
    LOG(
        (COUNT(CLOSE/DELAY(CLOSE,1) > ((CLOSE/DELAY(CLOSE,19))^(1/20)-1), 20) - 1) *
        SUMIF((CLOSE/DELAY(CLOSE,1) - ((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2, 20, 
              CLOSE/DELAY(CLOSE,1) < (CLOSE/DELAY(CLOSE,19))^(1/20)-1) /
        (COUNT((CLOSE/DELAY(CLOSE,1) < (CLOSE/DELAY(CLOSE,19))^(1/20)-1), 20) *
         SUMIF((CLOSE/DELAY(CLOSE,1) - ((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2, 20,
              CLOSE/DELAY(CLOSE,1) > (CLOSE/DELAY(CLOSE,19))^(1/20)-1))
    )
    
    逻辑：基于日收益率与20日年化收益率比较的方差比率检验
    """
    # 计算日收益率
    daily_ret = close / delay(close, 1) - 1
    
    # 计算20日年化收益率阈值：(CLOSE/DELAY(CLOSE,19))^(1/20)-1
    twenty_day_ratio = close / delay(close, 19)
    annualized_20d_ret = power(twenty_day_ratio, 1/20) - 1
    
    # 计算日收益率与年化收益率的偏离
    deviation = daily_ret - annualized_20d_ret
    squared_deviation = power(deviation, 2)
    
    # 条件1：日收益率 > 年化收益率
    cond_above = daily_ret > annualized_20d_ret
    # 条件2：日收益率 < 年化收益率  
    cond_below = daily_ret < annualized_20d_ret
    
    # 分子部分：
    # COUNT(条件1, 20) - 1
    count_above = count(cond_above, 20) - 1
    
    # SUMIF(平方偏离, 20, 条件2)
    sumif_below = sum_if(squared_deviation, 20, cond_below)
    
    numerator = count_above * sumif_below
    
    # 分母部分：
    # COUNT(条件2, 20)
    count_below = count(cond_below, 20)
    
    # SUMIF(平方偏离, 20, 条件1)
    sumif_above = sum_if(squared_deviation, 20, cond_above)
    
    denominator = count_below * sumif_above
    
    # 避免除零和无效值
    denominator = denominator.replace(0, np.nan)
    
    # 计算比率并取对数
    ratio = numerator / denominator
    result = log(ratio.replace(0, np.nan))
    
    shift = max(20, 19) - 1
    return result.iloc[shift:]

def alpha191(high, low, close, volume, **kwargs):
    """
    Alpha #191: ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
    """
    part1 = ts_corr(ts_mean(volume, 20), low, 5)
    part2 = (high + low) / 2
    result = (part1 + part2) - close
    shift = max(20, 5) - 1
    return result.iloc[shift:]

# ==================== 增强的因子计算器类 ====================

class AlphaCalculator191:
    """191因子计算器（增强版）"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        
        ###初始化因子计算器
        
        
        self.data = data_dict
        self.alpha_functions = self._get_alpha_functions()
    
    def _get_alpha_functions(self) -> Dict[str, callable]:
    
        return {
            'alpha100': alpha100,
            'alpha101': alpha101,
            'alpha102': alpha102,
            'alpha103': alpha103,
        'alpha104': alpha104,
        'alpha105': alpha105,
        'alpha106': alpha106,
        'alpha107': alpha107,
        'alpha108': alpha108,
        'alpha109': alpha109,
        'alpha110': alpha110,
        'alpha111': alpha111,
        'alpha112': alpha112,
        'alpha113': alpha113,
        'alpha114': alpha114,
        'alpha115': alpha115,
        'alpha116': alpha116,
        'alpha117': alpha117,
        'alpha118': alpha118,
        'alpha119': alpha119,
        'alpha120': alpha120,
        'alpha121': alpha121,
        'alpha122': alpha122,
        'alpha123': alpha123,
        'alpha124': alpha124,
        'alpha125': alpha125,
        'alpha126': alpha126,
        'alpha127': alpha127,
        'alpha128': alpha128,
        'alpha129': alpha129,
        'alpha130': alpha130,
        'alpha131': alpha131,
        'alpha132': alpha132,
        'alpha133': alpha133,
        'alpha134': alpha134,
        'alpha135': alpha135,
        'alpha136': alpha136,
        'alpha137': alpha137,
        'alpha138': alpha138,
        'alpha139': alpha139,
        'alpha140': alpha140,
        'alpha141': alpha141,
        'alpha142': alpha142,
        'alpha143': alpha143,
        'alpha144': alpha144,
        'alpha145': alpha145,
        'alpha146': alpha146,
        'alpha147': alpha147,
        'alpha148': alpha148,
        'alpha149': alpha149,
        'alpha150': alpha150,
        'alpha151': alpha151,
        'alpha152': alpha152,
        'alpha153': alpha153,
        'alpha154': alpha154,
        'alpha155': alpha155,
        'alpha156': alpha156,
        'alpha157': alpha157,
        'alpha158': alpha158,
        'alpha159': alpha159,
        'alpha160': alpha160,
        'alpha161': alpha161,
        'alpha162': alpha162,
        'alpha163': alpha163,
        'alpha164': alpha164,
        'alpha165': alpha165,
        'alpha166': alpha166,
        'alpha167': alpha167,
        'alpha168': alpha168,
        'alpha169': alpha169,
        'alpha170': alpha170,
        'alpha171': alpha171,
        'alpha172': alpha172,
        'alpha173': alpha173,
        'alpha174': alpha174,
        'alpha175': alpha175,
        'alpha176': alpha176,
        'alpha177': alpha177,
        'alpha178': alpha178,
        'alpha179': alpha179,
        'alpha180': alpha180,
        'alpha181': alpha181,
        'alpha182': alpha182,
        'alpha183': alpha183,
        'alpha184': alpha184,
        'alpha185': alpha185,
        'alpha186': alpha186,
        'alpha187': alpha187,
        'alpha188': alpha188,
        'alpha189': alpha189,
        'alpha190': alpha190,
        'alpha191': alpha191
        }
    
    
    def calculate_alpha(self, alpha_name: str) -> pd.DataFrame:
        """计算单个因子"""
        if alpha_name not in self.alpha_functions:
            raise ValueError(f"未知的因子名称: {alpha_name}")
        
        try:
            func = self.alpha_functions[alpha_name]
            result = func(**self.data)
            return result
        except Exception as e:
            print(f"计算因子 {alpha_name} 时出错: {e}")
            return pd.DataFrame()
    
    def calculate_all(self, alpha_list: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """批量计算因子"""
        if alpha_list is None:
            alpha_list = list(self.alpha_functions.keys())
        
        results = {}
        successful = 0
        failed = 0
        
        for alpha_name in alpha_list:
            result = self.calculate_alpha(alpha_name)
            if not result.empty:
                results[alpha_name] = result
                successful += 1
                print(f"✓ 成功计算因子: {alpha_name}")
            else:
                failed += 1
                print(f"✗ 计算因子 {alpha_name} 失败")
        
        print(f"\n计算完成！成功: {successful}, 失败: {failed}, 总计: {len(alpha_list)}")
        return results
    
    def get_available_alphas(self) -> List[str]:
        """获取可用的alpha因子列表"""
        return list(self.alpha_functions.keys())

# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 示例数据准备（需要根据实际情况调整）
    file_list = glob.glob(os.path.dirname('./') + './sp500_processedOHLC_CRSP/*.csv')
    code_list = list(map(lambda x: re.findall(r'\\(\w+).csv',x)[0], file_list))
    tw_kline = kline(code_list,start='20200101',target='us_stock')
    close_df = tw_kline.get_data('收盘')
    volume_df = tw_kline.get_data('成交量')
    high_df = tw_kline.get_data('最高')
    low_df = tw_kline.get_data('最低')
    #计算vwap
    open_df = tw_kline.get_data('开盘')
    vwap_df = (high_df + low_df + close_df + open_df) / 4
    # 计算returns (日收益率)
    returns_df = close_df.pct_change()
    # 计算adv数据 (平均成交量)
    adv10_df = ts_mean(volume_df, 10)
    adv20_df = ts_mean(volume_df, 20)
    adv30_df = ts_mean(volume_df, 30)
    adv40_df = ts_mean(volume_df, 40)
    adv50_df = ts_mean(volume_df, 50)
    adv60_df = ts_mean(volume_df, 60)
    adv150_df = ts_mean(volume_df, 150)
    adv180_df = ts_mean(volume_df, 180)
    
    # 生成示例数据
    data_dict = {
        'close': close_df,
        'open_price': open_df,
        'high': high_df,
        'low': low_df,
        'volume': volume_df,
        'vwap': vwap_df,
        'returns': returns_df,
        'adv10': adv10_df,
        'adv20': adv20_df,
        'adv30': adv30_df,
        'adv40': adv40_df,
        'adv50': adv50_df,
        'adv60': adv60_df,
        'adv150': adv150_df,
        'adv180': adv180_df,
        # 'cap': cap_df
        }
    
    
    # 创建因子计算器
    calculator = AlphaCalculator191(data_dict)
    
    '''
    # 计算因子100-191
    alpha_list_100_191 = [f'alpha{i:03d}' for i in range(100, 192)]
    print(f"\n开始计算因子100-191，共{len(alpha_list_100_191)}个因子...")
    results = calculator.calculate_all(alpha_list=alpha_list_100_191)
    
    # 显示结果示例
    for alpha_name, result_df in list(results.items())[:3]:
        print(f"\n{alpha_name} 结果示例:")
        print(result_df.iloc[-5:, :3])  # 显示最后5行，前3列
    '''

    # 创建输出目录
    output_dir = './factor_results_100_191_SP500'
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算因子100-191
    alpha_list_100_191 = [f'alpha{i:03d}' for i in range(100, 192)]
    print(f"\n开始计算因子100-191，共{len(alpha_list_100_191)}个因子...")
    
    # 添加异常处理的calculate_all方法
    results = {}
    results = calculator.calculate_all(alpha_list=alpha_list_100_191)
    failed_factors = []
    
    for alpha_name, result_df in list(results.items()):
        try:
            print(f"计算因子 {alpha_name}...")
            
            if result_df is not None and not result_df.empty:
                # 保存到CSV
                filename = f"{alpha_name}.csv"
                filepath = os.path.join(output_dir, filename)
                result_df.to_csv(filepath)
                results[alpha_name] = result_df
                print(f"  -> 已保存到 {filepath}")
            else:
                print(f"  -> {alpha_name} 计算结果为空，跳过保存")
                failed_factors.append(alpha_name)
                
        except Exception as e:
            print(f"  -> {alpha_name} 计算失败: {e}")
            failed_factors.append(alpha_name)
            continue
    
    # 生成计算报告
    print(f"\n{'='*50}")
    print("因子计算完成报告:")
    print(f"成功计算因子: {len(results)} 个")
    print(f"失败因子: {len(failed_factors)} 个")
    
    if failed_factors:
        print("失败的因子列表:")
        for factor in failed_factors:
            print(f"  - {factor}")
    
    # 显示部分成功因子的结果示例
    if results:
        print(f"\n前3个因子的结果示例:")
        for i, (alpha_name, result_df) in enumerate(list(results.items())[:3]):
            print(f"\n{alpha_name} 结果示例:")
            print(result_df.iloc[-5:, :3])  # 显示最后5行，前3列
    

