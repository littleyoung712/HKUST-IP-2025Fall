import numpy as np
import pandas as pd
from typing import Union

# 二元运算符cond?A:B，就是if(cond,A,B)，这里没有写了

def log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df)


def power(df: pd.DataFrame, exp: Union[float, int, pd.DataFrame]) -> pd.DataFrame:
    return df.pow(exp)


def rank(df: pd.DataFrame) -> pd.DataFrame:  # 截面排名
    return df.rank(axis=1, pct=True)





def scale(df: pd.DataFrame) -> pd.DataFrame:  # 截面标准化
    return df.div(df.abs().sum(axis=1), axis=0)


def sign(df: pd.DataFrame) -> pd.DataFrame:  # 符号
    return np.sign(df)

def delay(df:pd.DataFrame, span:int)->pd.DataFrame:
    # period阶滞后项
    return df.shift(span)

def delta(df:pd.DataFrame, span:int) -> pd.DataFrame:  # 时序差分
    return df.diff(span)

def ts_argmax(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # span内最大值对应的天数
    return df.rolling(span, min_periods=0).apply(np.argmax).add(1)


def ts_argmin(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # span内最小值对应的天数

    return df.rolling(span, min_periods=0).apply(np.argmin).add(1)


def ts_corr(df_1: pd.DataFrame, df_2: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序相关系数
    return df_1.rolling(span, min_periods=0).corr(df_2)


def ts_cov(df_1: pd.DataFrame, df_2: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序协方差
    return df_1.rolling(span, min_periods=0).cov(df_2)



def ts_max(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序最大
    return df.rolling(span, min_periods=0).max()


def ts_mean(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序均值
    return df.rolling(span, min_periods=0).mean()


def ts_min(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序最小

    return df.rolling(span, min_periods=0).min()





def ts_product(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序累乘
    return df.rolling(span, min_periods=0).apply(np.prod)


def ts_rank(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 最后一个值的时序排名
    return df.rolling(window=span, min_periods=0).apply(lambda x: x.rank().tail(1))


def ts_std(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序标准差
    return df.rolling(span, min_periods=0).std()


def ts_sum(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:  # 时序和
    return df.rolling(span, min_periods=0).sum()


def sma(df: pd.DataFrame, n: int, m: int) -> pd.DataFrame:  # 移动平均，SMA(x,n,m) = (m*x + (n-m)*SMA(x,n,m)_prev) / n
    alpha = m / n
    result = df.ewm(alpha=alpha).mean()
    return result


def decaylinear(df: pd.DataFrame, n: int) -> pd.DataFrame:  # 衰减求和
    # decaylinear(x,n) = (x[0] * n + x[1] * (n-1) + ... + x[n-1] * 1) / (n + (n-1) + ... + 1)
    weights = np.arange(n, 0, -1)
    weights_sum = weights.sum()

    def weighted_average(x):
        if len(x) < n:
            return np.nan
        return (x * weights[:len(x)]).sum() / weights_sum

    return df.rolling(n, min_periods=n).apply(weighted_average, raw=True)




def Regbeta(df: pd.DataFrame, x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:  # 回归取β
    window = len(x)
    return df.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0])


def Sumif(df: pd.DataFrame, span: int, cond) -> pd.DataFrame:  # 191似乎有，10没有
    df[~cond] = 0
    return df.rolling(span, min_periods=0).sum()




def Max(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:  # 对位最大
    return pd.DataFrame(data=np.maximum(df1, df2), index=df1.index, columns=df1.columns)


def Min(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:  # 对位最小
    return pd.DataFrame(data=np.minimum(df1, df2), index=df1.index, columns=df1.columns)





def Abs(df:pd.DataFrame)->pd.DataFrame:
    # 求绝对值
    return df.abs()





