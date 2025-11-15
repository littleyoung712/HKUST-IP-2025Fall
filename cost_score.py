import pandas as pd
import numpy as np
import talib

def calculate_bollinger_bands(close, period=20, dev=2):
    """
    布林带计算
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = sma + (std * dev)
    lower = sma - (std * dev)
    
    return upper, sma, lower

def calculate_atr(high, low, close, period=12):
    """
   实现ATR计算
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_cost_score(df, bandLimitBase=0.001, bandLimitMult=2.5, costAlloc=10, 
                        bb_period=20, bb_dev=2, atr_period=12):
    """
    计算cost score交易信号
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含OHLC数据，需要有列: 'open', 'high', 'low', 'close'
    bandLimitBase : float
        bandLimit的固定基础参数，默认0.001
    bandLimitMult : float
        bandLimit的ATR乘数，默认2.5
    costAlloc : float
        cost score的最大绝对值，默认10
    bb_period : int
        布林带周期，默认20
    bb_dev : float
        布林带标准差倍数，默认2
    atr_period : int
        ATR计算周期，默认12
        
    Returns:
    --------
    pandas.DataFrame
        包含date cost score的df
    """
    
    # result_df = df.sort_index()
    result_df = df.sort_values(by='date').reset_index(drop=True)
    
    # 1. 计算布林带
    # result_df['upperBound'], result_df['midBound'], result_df['lowerBound'] = talib.BBANDS(
    #     result_df['close'], timeperiod=bb_period, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    result_df['upperBound'], result_df['midBound'], result_df['lowerBound'] = calculate_bollinger_bands(
        result_df['close'], period=bb_period, dev=bb_dev)
    result_df['bandwidth'] =  result_df['upperBound'] -  result_df['lowerBound']

    # 2. 计算ATR
    # result_df['ATR'] = talib.ATR(result_df['high'], result_df['low'], result_df['close'], timeperiod=atr_period)
    result_df['ATR'] = calculate_atr(result_df['high'], result_df['low'], result_df['close'], period=atr_period)
    
    # 3. 计算bandWidth_pos_v
    close = result_df['close'].values
    lower = result_df['lowerBound'].values
    mid = result_df['midBound'].values
    upper = result_df['upperBound'].values
    bandwidth = result_df['bandwidth'].values

    bandWidth_pos_v = np.zeros_like(close)
    
    # 极度过卖区域: Close <= lowerBound
    mask_extreme_oversold = close <= lower
    bandWidth_pos_v[mask_extreme_oversold] = costAlloc
    
    # 弱势区域: lowerBound < Close <= midBound
    mask_weak_zone = (close > lower) & (close <= mid)
    bandWidth_pos_v[mask_weak_zone] = costAlloc * (mid[mask_weak_zone] - close[mask_weak_zone]) / (bandwidth[mask_weak_zone]/2)
    
    # 强势区域: midBound < Close <= upperBound
    mask_strong_zone = (close > mid) & (close <= upper)
    bandWidth_pos_v[mask_strong_zone] = -costAlloc * (close[mask_strong_zone]-mid[mask_strong_zone]) / (bandwidth[mask_strong_zone]/2)
    
    # 极度过买区域: Close > upperBound
    mask_extreme_overbought = close > upper
    bandWidth_pos_v[mask_extreme_overbought] = -costAlloc
    
    result_df['bandWidth_pos_v'] = bandWidth_pos_v
    
    # 4. 计算bandLimit
    result_df['bandLimit'] = np.maximum(
        bandLimitBase * result_df['close'], 
        result_df['ATR'] * bandLimitMult
    )
    
    # 5. 计算bandWidth_mult（带宽乘数）
    bb_width = result_df['upperBound'] - result_df['lowerBound']
    result_df['bandWidth_mult'] = np.minimum(bb_width / result_df['bandLimit'], 2)
    
    # 6. 计算最终的costScore
    raw_score = result_df['bandWidth_pos_v'] * result_df['bandWidth_mult']
    result_df['costScore'] = raw_score
    # 根据论文公式限制在±costAlloc范围内 
    pos_mask = result_df['bandWidth_pos_v'] > 0
    result_df['costScore'] = np.where(
        pos_mask,
        np.minimum(raw_score, costAlloc),
        np.maximum(raw_score, -costAlloc)
    )
    
    
    # 删除中间变量
    # final_columns = ['open', 'high', 'low', 'close', "volume", "return", "price_diff", 'costScore']
    final_columns = ['date', 'costScore']
    available_columns = [col for col in final_columns if col in result_df.columns]
    
    return result_df[available_columns]