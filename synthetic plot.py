import glob
import os
import re
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')


class Evaluation(object):
    def __init__(self, NV):
        self.NV = NV
        pass

    def get_annual_profit(self):  # 计算公式(1+r_year)^(day/365)=r
        days = (self.NV.index[-1] - self.NV.index[0]).days
        annual_profit = np.power(self.NV.iloc[-1] / self.NV.iloc[0], 1 / (days / 365)) - 1
        return annual_profit

    def get_max_withdraw(self, ):
        max_withdraw = 0
        max_withdraw_date = None
        peak = self.NV.iloc[0]
        i = 0
        for price in self.NV:
            if price > peak:
                peak = price
            withdraw = (peak - price) / peak
            if withdraw > max_withdraw:
                max_withdraw = withdraw
                max_withdraw_date = self.NV.index[i]
            i+=1

        return max_withdraw,max_withdraw_date

    def get_annual_volatility(self):
        # 计算每日收益率
        daily_returns = (self.NV - self.NV.shift(1)) / self.NV.shift(1)
        # 计算每日收益率的标准差，代表股票的日波动率
        daily_volatility = np.std(daily_returns)
        # 将日波动率转换为年波动率（假设一年有252个交易日）
        annual_volatility = daily_volatility * np.sqrt(252)
        return annual_volatility

    def get_sharpe(self):
        Er = self.get_annual_profit()-0.025
        sigma = self.get_annual_volatility()
        return Er / sigma

    def get_kamma(self):
        Er = self.get_annual_profit()
        withdraw,withdraw_date = self.get_max_withdraw()
        return Er / withdraw

    def generate_info(self):
        r = (self.NV.iloc[-1] - self.NV.iloc[0]) / self.NV.iloc[0]
        annual_r = self.get_annual_profit()
        sigma = self.get_annual_volatility()
        sharpe = self.get_sharpe()
        kamma = self.get_kamma()
        max_withdraw,max_withdraw_date = self.get_max_withdraw()
        return pd.Series(data=[r, annual_r, sigma, max_withdraw,max_withdraw_date, sharpe, kamma],
                         index=['区间收益率', '年化收益率', '年化波动率', '最大回撤','最大回撤日期' ,'夏普比率', '卡玛比率'])



def load_factor_nv_files(base_dir: str, target: str) -> Dict[str, pd.DataFrame]:
    """
    读取指定市场所有因子回测净值文件
    """
    target_path = os.path.join(base_dir, target)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"路径不存在: {target_path}")

    factor_files = glob.glob(os.path.join(target_path, '*', 'Factor_NV.csv'))
    factor_data = {}

    for file in factor_files:
        factor_name = re.findall(rf'{re.escape(target_path)}\\(.+?)\\Factor_NV.csv', file)
        if not factor_name:
            continue
        factor_name = factor_name[0]
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            factor_data[factor_name] = df
        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")

    if not factor_data:
        raise ValueError(f"未读取到任何Factor_NV文件, 检查路径: {target_path}")

    return factor_data





def compute_factor_metrics(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if 'Long-Short' not in df.index and 'Long-Short' not in df.columns:
        raise KeyError('Factor_NV中未找到Long-Short列/行')

    # 兼容行列格式
    if 'Long-Short' in df.index:
        nv_series = df.loc['Long-Short']
    else:
        nv_series = df['Long-Short']

    nv_series.index = pd.to_datetime(nv_series.index)
    EV = Evaluation(nv_series)

    metrics = {
        'autocorr_lag1': (nv_series.pct_change(1).autocorr(lag=1)),
        'annual_return': EV.get_annual_profit(),
        'max_drawdown': -EV.get_max_withdraw()[0],
        'annual_volatility': -EV.get_annual_volatility(),
        'sharpe_ratio': EV.get_sharpe(),
    }

    return nv_series, pd.Series(metrics)








def select_top_factors(base_dir: str, target: str = 'us_stock', k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame]:
    factor_data = load_factor_nv_files(base_dir, target)

    nv_collection = {}
    metrics_records = {}

    for factor_name, df in factor_data.items():
        try:
            nv_series, metrics = compute_factor_metrics(df)
            nv_collection[factor_name] = nv_series
            metrics_records[factor_name] = metrics
        except Exception as e:
            print(f"因子 {factor_name} 处理失败: {e}")
            continue

    if not nv_collection:
        raise ValueError('未能成功计算任何因子')

    metrics_df = pd.DataFrame(metrics_records).T
    scores_df = metrics_df.rank(pct=True,axis=0)
    scores_df['final_score'] = scores_df.mean(axis=1)
    top_factors = scores_df.sort_values('final_score', ascending=False).head(k).index
    combined_long_short = pd.concat({name: nv_collection[name] for name in top_factors}, axis=1)

    return combined_long_short, scores_df.loc[top_factors],metrics_df.loc[top_factors]


if __name__ == '__main__':
    res_dir = r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子回测结果'
    a,b,c = select_top_factors(res_dir,'us_stock',20)
    a[b.index].plot()
    plt.show(block=True)
