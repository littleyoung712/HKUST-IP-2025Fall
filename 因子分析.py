import glob
import re

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from datetime import datetime as dt
import warnings
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from typing import (
    List,
    Optional,
    Tuple,
)
from multiprocessing import Pool
start_date = '20200101'  # 每次需要调整上面的部分
end_date = '20250122'
mkt_neutral = False
use_constitution = False
constitute = 'zz800'  # 每次需要调整
prefix = '全A'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class kline(object):  # 返回
    Data = None

    def __init__(self, codes: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None,
                 target: str = 'us_stock'):
        # target有两种，us_stock(美股), tw_stock(台股),
        self.target = target
        trading_days = self.get_trading_days(start, end, target)
        self.indicator_names = None
        self.Data = self.load_data(trading_days, codes)

    def __getitem__(self, date: pd.Timestamp) -> pd.DataFrame:
        return self.Data.loc[date]

    @staticmethod
    def get_trading_days(start: Optional[str] = None, end: Optional[str] = None, target: str = 'us_stock') -> Optional[List[pd.Timestamp]]:
        # 根据target类型选择因子文件路径
        if target == 'us_stock':
            factor_file = r'D:\大学文档\MAFM\IP\AtrDataCode\merged_factors\美股因子.csv'
        else :
            factor_file = r'D:\大学文档\MAFM\IP\AtrDataCode\merged_factors\台股因子.csv'

        # 从因子文件中读取交易日期
        if os.path.exists(factor_file):
            try:
                # 读取因子文件，假设有date列
                factor_df = pd.read_csv(factor_file, parse_dates=['date'], encoding='utf-8')
                trading_days = pd.to_datetime(factor_df['date']).sort_values().unique()
                
                # 根据start和end参数筛选日期
                if start:
                    start_date = pd.to_datetime(start)
                    trading_days = trading_days[trading_days >= start_date]
                
                if end:
                    end_date = pd.to_datetime(end)
                    trading_days = trading_days[trading_days <= end_date]
                
                return trading_days.tolist()
            except Exception as e:
                print(f"从{target}因子文件提取交易日期时出错: {e}")
                # 如果失败，回退到A股方法
                pass
        else:
            print(f"因子文件不存在: {factor_file}")
            # 如果失败，回退到A股方法
            pass
        





    def load_data(self, dates: List[pd.Timestamp], codes: Optional[List[str]] = None) -> pd.DataFrame:
        # 根据target类型选择数据路径
        if self.target == 'us_stock':
            # 美股数据路径
            data_path = r'D:\大学文档\MAFM\IP\AtrDataCode\sp500_processedOHLC_CRSP'
        elif self.target == 'tw_stock':
            # 台股数据路径
            data_path = r'D:\大学文档\MAFM\IP\AtrDataCode\tw50_processedOHLC'

        else:
            raise ValueError('target必须为 stock, tw_stock其中一种！')
        
        if self.target in ['us_stock', 'tw_stock']:
            # 美股和台股数据处理
            import glob
            csv_files = glob.glob(os.path.join(data_path, '*.csv'))
            
            if not csv_files:
                raise FileNotFoundError(f"在路径 {data_path} 下未找到CSV文件")
            
            # 读取所有CSV文件并合并
            all_data = []
            for file in csv_files:
                try:
                    if self.target == 'us_stock':
                        # 美股数据处理
                        df = pd.read_csv(file, index_col=0, parse_dates=['DATE'], encoding='utf-8')
                        # 重命名列以匹配原始格式
                        df = df.rename(columns={
                            'DATE': '日期',
                            'Open': '开盘',
                            'High': '最高',
                            'Low': '最低',
                            'Close': '收盘',
                            'Volume': '成交量'
                        })
                    else:  # tw_stock
                        # 台股数据处理
                        df = pd.read_csv(file, index_col=None, parse_dates=['date'], encoding='utf-8')
                        # 重命名列以匹配原始格式
                        df = df.rename(columns={
                            'date': '日期',
                            'Adj_Open': '开盘',
                            'Adj_High': '最高',
                            'Adj_Low': '最低',
                            'Adj_Close': '收盘',
                            'Adj_Volume': '成交量'
                        })
                    
                    # 添加股票代码列（从文件名提取）
                    stock_code = os.path.basename(file).replace('.csv', '')
                    df['代码'] = stock_code
                    all_data.append(df)
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {e}")
                    continue
            
            if not all_data:
                raise ValueError("未能成功读取任何CSV文件")
            
            # 合并所有数据
            data = pd.concat(all_data, ignore_index=False)
            data = data.reset_index()
            
            # 筛选指定日期范围
            if dates:
                data = data[data['日期'].isin(dates)]
            
            # 筛选指定股票代码
            if codes:
                data = data[data['代码'].isin(codes)]
            
            # 设置指标名称
            self.indicator_names = ['开盘', '最高', '最低', '收盘', '成交量']
            
        else:
            # 原有的A股数据处理逻辑
            target_name = re.findall(r'(.*)日', data_path)[0] + '代码'
            data = pd.read_csv(data_path, index_col=0, parse_dates=True, encoding='gbk')
            data = data.loc[dates]
            if codes:
                data = data[data[target_name].isin(codes)]

            data = data.reset_index().rename(columns={'level0': '日期', target_name: '代码'})
            self.indicator_names = data.columns[(data.columns != '日期') & (data.columns != '代码')].tolist()
        
        return data

    def get_data(self, indicator: str) -> pd.DataFrame:
        if indicator in self.indicator_names:
            pivot_data = pd.pivot_table(data=self.Data, values=indicator, index='日期', columns='代码')
            return pivot_data
        else:
            raise ValueError(f"指标名必须为 {' '.join(self.indicator_names)}")


class Factor_Analysis(object):  # change your save_path
    POOL = 'A Share'
    shift_dict = {'D': 1, 'W': 5, 'M': 21, 'Y': 252}

    def __init__(self, factor_df: pd.DataFrame, factor_name: Optional[str] = None, codes: Optional[List[str]] = None,
                 start: Optional[str] = None,
                 end: Optional[str] = None, freq: str = 'D', target: str = 'us_stock'):
        self.target = target
        self.profit = None
        self.hierarchical_constitute = None  # 后freq日购买的股票
        self.str_freq = freq
        self.freq = self.shift_dict[freq]
        self.codes = codes
        self.BENCHMARK_PATH = "D:\大学文档\MAFM\IP\AtrDataCode\merged_factors\SPX.xlsx" if target=='us_stock' else "D:\大学文档\MAFM\IP\AtrDataCode\merged_factors\TWII.xlsx"
        self.kline = kline(codes, start, end, target=target)
        self.trading_day = pd.to_datetime(kline.get_trading_days(start, end,target=target))
        self.trading_day = self.trading_day[self.trading_day.isin(factor_df.index)][
                           :-self.freq]  #  最后N天的因子不用，因为回测用下N日数据
        self.dates = self.backtest_dates()
        self.factor_df = self.align(original_factor=factor_df, dates=self.trading_day.tolist(), codes=self.codes)
        self.std_factor_df = self.preprocess()
        self.r_next = self.align(self.get_r_next(), dates=self.trading_day.tolist(),
                                 codes=self.std_factor_df.columns.tolist())  # 只提取有因子数据的
        self.trade_state_next = self.align(self.get_trading_state_next(), dates=self.trading_day.tolist(),
                                           codes=self.std_factor_df.columns.tolist())  # 只提取有因子数据的
        self.benchmark_r_next = pd.read_excel(self.BENCHMARK_PATH,
                                              index_col=1, parse_dates=True)['收盘价'].pct_change(
            periods=self.freq).shift(-self.freq).dropna()
        self.name = factor_name

    @staticmethod
    def three_sigma(cs_factor: pd.Series) -> pd.Series:  # 截面去极值

        mean = np.nanmean(cs_factor)
        std = np.nanstd(cs_factor)
        cs_factor = cs_factor.loc[
            (cs_factor > mean - 3 * std) & (cs_factor < mean + 3 * std)]
        return cs_factor

    @staticmethod
    def align(original_factor: pd.DataFrame, dates: Optional[List[pd.Timestamp]] = None,
              codes: Optional[List[str]] = None) -> pd.DataFrame:  # 只选用在池子内，且在交易日期内的factor
        use_factor = original_factor
        if dates is not None:
            use_factor = use_factor.reindex(index=dates)
        if codes is not None:
            use_factor = use_factor.reindex(columns=codes)
        return use_factor

    def get_trading_state_next(self) -> pd.DataFrame:
        high = self.kline.get_data('最高')
        low = self.kline.get_data('最低')
        low.index, high.index = pd.to_datetime(
            low.index), pd.to_datetime(low.index)
        trade_permission = high != low
        trade_permission[trade_permission == -1] = 1
        trade_permission[trade_permission != 1] = 0
        trade_state_next = (trade_permission.shift(-self.freq)).dropna(how='all', axis=0)
        return trade_state_next

    def get_r_next(self) -> pd.DataFrame:
        close_adj = self.kline.get_data('收盘')
        zero_close = np.where(close_adj == 0)
        if len(zero_close) > 0:
            zero_close_positions = [(zero_close[0][i], zero_close[1][i]) for i in range(len(zero_close[0]))]
            for position in zero_close_positions:
                close_adj.iloc[position] = close_adj.iloc[position[0] - 1, position[1]]
        r_adj = close_adj.pct_change(periods=self.freq)
        r_adj.index = pd.to_datetime(r_adj.index)
        return (r_adj.shift(-self.freq)).dropna(how='all', axis=0)

    def standard(self, cs_factor: pd.Series, type: Optional[str] = 'Minmax') -> pd.Series:  # 截面标准化
        cs_factor = self.three_sigma(cs_factor)
        if type == 'Minmax':
            col_min = np.nanmin(cs_factor)
            col_max = np.nanmax(cs_factor)
            cs_factor = (cs_factor - col_min) / (col_max - col_min)
            return cs_factor
        elif type == 'Normal':
            mean = np.nanmean(cs_factor)
            std = np.nanstd(cs_factor)
            cs_factor = (cs_factor - mean) / std
            return cs_factor

    def preprocess(self) -> pd.DataFrame:  # 因子截面标准化+去极值
        standard_factor = self.factor_df.apply(lambda x: self.standard(cs_factor=x,type = 'Normal'), axis=1).dropna(axis=1, how='all')
        return standard_factor

    def backtest_dates(self) -> List[pd.Timestamp]:
        if self.freq == 1:
            date_range = self.trading_day
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(days=1)]
        elif self.freq == 5:
            date_range = self.trading_day[::5]
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(weeks=1)]
        elif self.freq == 21:
            date_range = self.trading_day[::21]
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(months=1)]
        else:
            date_range = self.trading_day[::252]
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(years=1)]
        return profit_dates

    def st_filter(self, date: pd.Timestamp) -> pd.Index:
        trade_state = self.trade_state_next.loc[date]
        trading_stock = trade_state[trade_state == 1].index
        return trading_stock

    def Stratify_r(self, date: pd.Timestamp) -> Tuple:  # 返回分层收益率(下一交易日)和分层成分
        print(date)
        trading_stock = self.st_filter(date)
        daily_factor = self.std_factor_df.loc[date]
        daily_next_r = pd.DataFrame(self.r_next.loc[date, trading_stock])
        factor_quantiles = daily_factor.quantile([0, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1])
        dup_pos_list = np.where(factor_quantiles.duplicated())[0]
        if len(dup_pos_list) >=3:
            print('Data too concentrated!')
            division_spot = np.linspace(0,len(daily_factor),6).astype('int')
            st_layer = pd.cut(np.arange(0,len(daily_factor)), bins=division_spot,
                              labels=['1', '2', '3', '4', '5'],
                              include_lowest=True, right=True)
            st_layer = pd.Series(data=st_layer.tolist(),index=daily_factor.sort_values(ascending=True).index)
        elif len(factor_quantiles) != len(factor_quantiles.drop_duplicates()):
            if dup_pos_list[-1]==len(factor_quantiles)-1:
                print('last layer concentrated!')
                division_spot = np.linspace(0, len(daily_factor), 6).astype('int')
                st_layer = pd.cut(np.arange(0, len(daily_factor)), bins=division_spot,
                                  labels=['1', '2', '3', '4', '5'],
                                  include_lowest=True, right=True)
                st_layer = pd.Series(data=st_layer.tolist(), index=daily_factor.sort_values(ascending=True).index)
            else:
                print('Data concentrated！')
                old_pos = 0
                i = 0
                while len(dup_pos_list) != 0:
                    dup_pos = dup_pos_list[i]
                    if dup_pos == old_pos:
                        i += 1
                    old_dup_pos_list= dup_pos_list.copy()
                    factor_quantiles.iloc[dup_pos] = (factor_quantiles.iloc[dup_pos] + factor_quantiles.iloc[dup_pos + 1]) / 2
                    dup_pos_list = np.where(factor_quantiles.duplicated())[0]
                    if len(old_dup_pos_list) != len(dup_pos_list):
                        i=0
                    old_pos = dup_pos
                st_layer = pd.cut(daily_factor, bins=factor_quantiles,
                                  labels=['1', '2', '3', '4', '5'],
                                  include_lowest=True, right=True)
        else:
            st_layer = pd.cut(daily_factor, bins=factor_quantiles,
                              labels=['1', '2', '3', '4', '5'],
                              include_lowest=True, right=True)




        daily_next_r['level'] = st_layer.loc[daily_next_r.index]
        layer_return = daily_next_r.groupby(by='level').apply(np.nanmean).fillna(0)

        layer_return.loc['1-5'] = layer_return.loc['1'] - layer_return.loc['5']
        layer_return.loc['5-1'] = layer_return.loc['5'] - layer_return.loc['1']

        return layer_return, daily_next_r['level']

    def Period_NV_Backtest(self) -> pd.DataFrame:  # 返回一段期间的给定name的净值变化

        period_factor_NV = pd.DataFrame()
        hierarchical_constitute = pd.DataFrame()
        for date in self.trading_day:
            factor_r, daily_constitute = self.Stratify_r(date)  # 后一日的收益率
            factor_NV = factor_r + 1
            period_factor_NV = pd.concat([period_factor_NV, factor_NV], axis=1)
            hierarchical_constitute = pd.concat([hierarchical_constitute, daily_constitute], axis=1)
        period_factor_NV.columns = self.trading_day
        hierarchical_constitute.columns = self.trading_day
        self.hierarchical_constitute = hierarchical_constitute
        return period_factor_NV  # 返回的格式是单因子分层的时序区间收益率

    def Backtest(self):  # 计算区间内的单因子回测情况
        NV = self.Period_NV_Backtest()
        profit = pd.DataFrame(index=NV.index,
                              data=np.ones([len(NV.index), 1]))
        date_range = self.trading_day[::self.freq]
        for date in date_range:
            select_NV = NV[date]
            profit = pd.concat([profit, profit.iloc[:, -1] * select_NV], axis=1)
        profit.columns = self.dates
        label = '1-5' if profit.iloc[-2, -1] > profit.iloc[-1, -1] else '5-1'
        profit.loc['Long-Short', :] = profit.loc[label, :]
        self.profit = profit
        return profit

    def Rank_IC(self) -> pd.Series:
        Rank_IC_sr = self.std_factor_df.corrwith(self.r_next, method='spearman', axis=1)
        return Rank_IC_sr

    def IC(self) -> pd.Series:
        IC_sr = self.std_factor_df.corrwith(self.r_next, method='pearson', axis=1)
        return IC_sr

    def turnover(self) -> pd.DataFrame:
        """
        计算换手率：相邻两期每层权重变化的绝对值之和 / 2
        假设所有股票均等权组合，即每只股票权重 = 1 / 该层股票总数
        """
        date_range = self.trading_day[::self.freq]
        turnover_high_list = [0]
        turnover_low_list = [0]
        
        for i in range(len(date_range) - 1):
            position_old = self.hierarchical_constitute[date_range[i]]
            position_new = self.hierarchical_constitute[date_range[i + 1]]
            
            # 获取每层的股票集合
            position_old_high = set(position_old[position_old == '5'].index.tolist())
            position_old_low = set(position_old[position_old == '1'].index.tolist())
            position_new_high = set(position_new[position_new == '5'].index.tolist())
            position_new_low = set(position_new[position_new == '1'].index.tolist())
            
            # 计算各层股票数量，避免除零
            n_old_high = len(position_old_high) if len(position_old_high) > 0 else 1
            n_old_low = len(position_old_low) if len(position_old_low) > 0 else 1
            n_new_high = len(position_new_high) if len(position_new_high) > 0 else 1
            n_new_low = len(position_new_low) if len(position_new_low) > 0 else 1
            
            # 初始化权重字典
            weight_old_high = {stock: 1.0 / n_old_high for stock in position_old_high}
            weight_old_low = {stock: 1.0 / n_old_low for stock in position_old_low}
            weight_new_high = {stock: 1.0 / n_new_high for stock in position_new_high}
            weight_new_low = {stock: 1.0 / n_new_low for stock in position_new_low}
            
            # 计算两期的权重差（绝对值求和）
            all_stocks_high = position_old_high.union(position_new_high)
            all_stocks_low = position_old_low.union(position_new_low)
            
            turnover_high = sum(abs(weight_new_high.get(stock, 0) - weight_old_high.get(stock, 0)) 
                               for stock in all_stocks_high) / 2
            turnover_low = sum(abs(weight_new_low.get(stock, 0) - weight_old_low.get(stock, 0)) 
                              for stock in all_stocks_low) / 2
            
            turnover_high_list.append(turnover_high)
            turnover_low_list.append(turnover_low)
        
        turnover_high_sr = pd.Series(data=turnover_high_list, index=self.dates[1:])
        turnover_low_sr = pd.Series(data=turnover_low_list, index=self.dates[1:])
        turnover_df = pd.concat([turnover_high_sr, turnover_low_sr], axis=1)
        turnover_df.columns = ['layer5 turnover', 'layer1 turnover']
        return turnover_df

    def hit_rate(self) -> pd.DataFrame:
        date_range = self.trading_day[::self.freq]
        layer_1_hit_list = []
        layer_5_hit_list = []
        for i in range(len(date_range)):
            daily_cons = self.hierarchical_constitute[date_range[i]]
            daily_return = self.r_next.loc[date_range[i]]
            layer_1_cons = daily_cons[daily_cons == '1']
            layer_5_cons = daily_cons[daily_cons == '5']
            benchmark_return = self.benchmark_r_next.loc[date_range[i]]
            layer_1_return = daily_return.loc[layer_1_cons.index]
            layer_5_return = daily_return.loc[layer_5_cons.index]
            layer_5_rate =  (layer_5_return > benchmark_return).value_counts(1)
            layer_1_rate = (layer_1_return > benchmark_return).value_counts(1)
            layer_5_hit_rate = layer_5_rate.loc[True] if True in layer_5_rate.index else 0
            layer_1_hit_rate = layer_1_rate.loc[True] if True in layer_1_rate.index else 0
            layer_1_hit_list.append(layer_1_hit_rate)
            layer_5_hit_list.append(layer_5_hit_rate)
        hit_high_sr = pd.Series(data=layer_5_hit_list, index=self.dates[1:])
        hit_low_sr = pd.Series(data=layer_1_hit_list, index=self.dates[1:])
        hit_df = pd.concat([hit_high_sr, hit_low_sr], axis=1)
        hit_df.columns = ['layer5 hit_rate', 'layer1 hit_rate']
        hit_df -= 0.5
        return hit_df

    def Backtest_Plot(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        Period_profit = self.profit
        Period_profit.loc[['1', '2', '3', '4', '5', 'Long-Short']].T.plot()
        plt.legend()
        plt.title('{} Factor Performance'.format(self.name))
        save_dir = os.path.join(r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子分层表现',
                         f'{self.target}',self.str_freq, f'{self.name}')
        if not os.path.exists(save_dir):
            make_dir(save_dir)
        plt.savefig(os.path.join(save_dir
            ,
                         f'start={dt.strftime(self.trading_day[0], "%Y%m%d")} freq={self.freq} 分层回测.png'))
        plt.close()

    def line_plot(self, data: pd.Series, figure_name: str = 'IC'):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        ax.plot(data.index, data.values, label='{}'.format(figure_name), lw=1)
        ax.text(.05, .95, "Mean %.3f \n Std. %.3f" % (data.mean(), data.std()),
                fontsize=16,
                bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=ax.transAxes,
                verticalalignment='top')
        ax.set(xlabel="", title='{} {} Factor {} Distribution'.format(self.POOL, self.name, figure_name))
        ax.set_ylabel('{}'.format(figure_name), color='blue')

        ax.tick_params(axis='y', labelcolor='blue')
        cum_data = data.cumsum()
        ax1 = ax.twinx()
        ax1.set_ylabel('Cumulative {}'.format(figure_name), color='red')
        ax1.plot(data.index, cum_data, color='red', ls='-', alpha=0.8, label='Cumulative {}'.format(figure_name))
        ax1.tick_params(axis='y', labelcolor='red')
        fig.legend(loc='upper right')
        save_dir = os.path.join(r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子分层表现',
                         f'{self.target}',self.str_freq, f'{self.name}')
        if not os.path.exists(save_dir):
            make_dir(save_dir)
        plt.savefig(os.path.join(save_dir,

                         f'start={dt.strftime(self.trading_day[0], "%Y%m%d")} freq={self.freq} {figure_name}.png'))
        plt.close()



    def bar_plot(self, figure_name: str = 'turnover'):
        function_dict = {'turnover': self.turnover(), 'hit_rate': self.hit_rate()}
        try:
            df = function_dict[figure_name]
            save_dir = os.path.join(r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子回测结果', f'{self.target}',self.str_freq,
                                    f'{self.name}')
            if not os.path.exists(save_dir):
                make_dir(save_dir)
            df.to_csv(save_dir + rf'\{figure_name}.csv', encoding='gbk')
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].bar(df.index, df[f'layer5 {figure_name}'], width=10)
            ax[0].set_title(f'Layer 5 {figure_name}', fontsize=16)
            ax[1].bar(df.index, df[f'layer1 {figure_name}'], width=10)
            ax[1].set_title(f'Layer 1 {figure_name}', fontsize=16)
            save_dir = os.path.join(r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子分层表现',self.str_freq,
                                     f'{self.target}', f'{self.name}')
            if not os.path.exists(save_dir):
                make_dir(save_dir)
            plt.savefig(os.path.join(save_dir
                                     ,
                             f'start={dt.strftime(self.trading_day[0], "%Y%m%d")} freq={self.freq} {figure_name}.png'))
            # plt.show(block=True)
        except Exception as e:
            print('Encounter error:', e)
            print('Figure name must be one of "turnover" and "hit_rate"')

    def factor_return(self) -> pd.Series:

        factor_returns = []
        for date in self.trading_day:
            trading_stock = self.st_filter(date)
            daily_factor = self.std_factor_df.loc[date, trading_stock].dropna()
            daily_next_r = self.r_next.loc[date, daily_factor.index]
            daily_factor = sm.add_constant(daily_factor)
            # 计算因子收益率
            factor_model = sm.OLS(daily_next_r.values, daily_factor.values).fit()
            factor_return = factor_model.params[1]
            factor_returns.append(factor_return)
            
        factor_return_series = pd.Series(factor_returns, index=self.trading_day)
        return factor_return_series

    def save_data(self):
        save_dir = os.path.join(r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子回测结果',f'{self.target}',self.str_freq, f'{self.name}')
        if not os.path.exists(save_dir):
            make_dir(save_dir)
        self.profit.to_csv(save_dir+ '\Factor_NV.csv', encoding='gbk')
        self.hierarchical_constitute.to_csv(save_dir+ '\Hierarchical_Holding.csv', encoding='gbk')


    def comprehensive_analysis_plot(self):
        """
        生成综合分析图：包含IC、Rank_IC、分层折线图、分层年化柱状图、turnover和hitrate
        布局为2*3
        IC和Rank_IC调用line_plot的逻辑，turnover和hitrate调用bar_plot的逻辑
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建2*3的子图
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 计算所需数据
        IC_data = self.IC()
        Rank_IC_data = self.Rank_IC()
        turnover_df = self.turnover()
        hit_rate_df = self.hit_rate()
        
        # 计算年化收益率
        profit_df = self.profit
        days_per_year = 252/(self.trading_day[-1]-self.trading_day[0]).days
        annualized_return = profit_df.loc[['1', '2', '3', '4', '5'], self.dates].iloc[:, -1] **days_per_year  - 1
        
        # 1. IC图 (0,0) - 使用line_plot的逻辑
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(IC_data.index, IC_data.values, label='IC', lw=1, color='blue')
        cum_data_ic = IC_data.cumsum()
        ax1_twin = ax1.twinx()
        ax1_twin.plot(IC_data.index, cum_data_ic, color='red', ls='-', alpha=0.8, label='Cumulative IC')
        ax1.text(.05, .95, "Mean %.3f \n Std. %.3f" % (IC_data.mean(), IC_data.std()),
                fontsize=12, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=ax1.transAxes, verticalalignment='top')
        ax1.set_title(f'{self.POOL} {self.name} Factor IC Distribution', fontsize=14)
        ax1.set_ylabel('IC', fontsize=10, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.set_ylabel('Cumulative IC', fontsize=10, color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)
        ax1_twin.legend(loc='upper right', fontsize=8)
        
        # 2. Rank_IC图 (0,1) - 使用line_plot的逻辑
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(Rank_IC_data.index, Rank_IC_data.values, label='Rank_IC', lw=1, color='green')
        cum_data_rankic = Rank_IC_data.cumsum()
        ax2_twin = ax2.twinx()
        ax2_twin.plot(Rank_IC_data.index, cum_data_rankic, color='red', ls='-', alpha=0.8, label='Cumulative Rank_IC')
        ax2.text(.05, .95, "Mean %.3f \n Std. %.3f" % (Rank_IC_data.mean(), Rank_IC_data.std()),
                fontsize=12, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=ax2.transAxes, verticalalignment='top')
        ax2.set_title(f'{self.POOL} {self.name} Factor Rank_IC Distribution', fontsize=14)
        ax2.set_ylabel('Rank_IC', fontsize=10, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2_twin.set_ylabel('Cumulative Rank_IC', fontsize=10, color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=8)
        ax2_twin.legend(loc='upper right', fontsize=8)
        
        # 3. 分层折线图 (0,2)
        ax3 = fig.add_subplot(gs[0, 2])
        profit_df.loc[['1', '2', '3', '4', '5', 'Long-Short']].T.plot(ax=ax3, linewidth=1.5)
        ax3.set_title('Factor Performance by Layer', fontsize=14)
        ax3.set_ylabel('Cumulative Return', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # 4. 分层年化柱状图 (1,0)
        ax4 = fig.add_subplot(gs[1, 0])
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
        bars = ax4.bar(annualized_return.index, annualized_return.values, color=colors, alpha=0.8)
        ax4.set_title('Annualized Return by Layer', fontsize=14)
        ax4.set_ylabel('Annualized Return', fontsize=10)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 5. Turnover图 (1,1) - 使用bar_plot的逻辑
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(turnover_df.index, turnover_df['layer1 turnover'], label='Layer 1 turnover', width=10, alpha=0.8)
        ax5.text(.05, .95, f"Layer1 Mean: {turnover_df['layer1 turnover'].mean():.3f}",
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=ax5.transAxes, verticalalignment='top')
        ax5.set_title('Turnover Rate', fontsize=14)
        ax5.set_ylabel('Turnover', fontsize=10)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Hit Rate图 (1,2) - 使用bar_plot的逻辑
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(hit_rate_df.index, hit_rate_df['layer1 hit_rate'], label='Layer 1 hit_rate', width=10, alpha=0.8)
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.text(.05, .95, f"Layer1 Mean: {hit_rate_df['layer1 hit_rate'].mean():.3f}",
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=ax6.transAxes, verticalalignment='top')
        ax6.set_title('Hit Rate vs Benchmark', fontsize=14)
        ax6.set_ylabel('Hit Rate (centered)', fontsize=10)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 设置整体标题
        fig.suptitle(f'{self.name} Factor Comprehensive Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # 保存图片
        save_dir = os.path.join(r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子分层表现',f'{self.target}综合分析',self.str_freq)
        if not os.path.exists(save_dir):
            make_dir(save_dir)
        save_path = os.path.join(save_dir,f'{self.name}_综合分析_开始于{dt.strftime(self.trading_day[0], "%Y%m%d")}_frequency={self.freq}.png')

        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"综合分析图已保存至: {save_path}")


def neutralization(factor_df, name, mkt_cap=False, industry=False):  # 这一部分还没写好，需要每一个因子每天的市值以及所属的行业,两个pivot类型的
    neutralized_factor = []
    if type(mkt_cap) == pd.DataFrame:  # 对界面进行中性化
        lg_mkt_cap = mkt_cap.apply(lambda x: np.log(x))
        if type(industry) == pd.DataFrame:  # 行业、市值
            for date in tqdm(factor_df.index):
                factor_sr = factor_df.loc[date]
                industry_sr = industry.loc[date]
                lg_mkt_sr = lg_mkt_cap.loc[date]
                test_df = pd.concat([factor_sr, lg_mkt_sr, industry_sr], axis=1)
                test_df.columns = [name, 'Ln_mkt_cap', 'industry']
                result = smf.ols(f'{name}~Ln_mkt_cap+C(industry)', test_df).fit()
                res = result.resid
                neutralized_factor.append(res)

        else:  # 仅市值
            for date in tqdm(factor_df.index):
                factor_sr = factor_df.loc[date]
                lg_mkt_sr = lg_mkt_cap.loc[date]
                test_df = pd.concat([factor_sr, lg_mkt_sr], axis=1)
                test_df.columns = [name, 'Ln_mkt_cap']
                result = smf.ols(f'{name}~Ln_mkt_cap', test_df).fit()
                res = result.resid
                neutralized_factor.append(res)
    elif type(industry) == pd.DataFrame:  # 仅行业
        for date in tqdm(factor_df.index):
            factor_sr = factor_df.loc[date]
            industry_sr = industry.loc[date]
            test_df = pd.concat([factor_sr, industry_sr], axis=1)
            test_df.columns = [name, 'industry']
            result = smf.ols(f'{name}~C(industry)', test_df).fit()
            res = result.resid
            neutralized_factor.append(res)
    neutralized_factor_df = pd.concat(neutralized_factor, axis=1).T
    neutralized_factor_df.index = factor_df.index
    return neutralized_factor_df


def single_factor_test(factor_df, col, freq='D',target='us_stock', start='20200101', end='20250101'):  # freq设置D,W,M,Y，即回测频率
    try:
        backtester = Factor_Analysis(factor_df, col, start=start, end=end, freq=freq, target=target)
        strategy_r = backtester.Backtest()
        backtester.Backtest_Plot()
        IC = backtester.IC()
        Rank_IC = backtester.Rank_IC()
        factor_rtn = backtester.factor_return()
        backtester.line_plot(IC, 'IC')
        backtester.line_plot(Rank_IC, 'Rank_IC')
        backtester.line_plot(factor_rtn, 'factor_rtn')
        backtester.bar_plot('turnover')
        backtester.bar_plot('hit_rate')
        backtester.comprehensive_analysis_plot()
        backtester.save_data()

        return strategy_r
    except Exception as e:
        print(f'因子{col}回测出错:{e}')


def select_constitute(factor, cons):  # 输入我的数据以及成分股，成分股是pivot类型的，因子数据也是pivot类型的,输出的是双重索引的unstack
    daily_cons = pd.DataFrame(cons.unstack().dropna())
    daily_cons = daily_cons.rename(columns={0: 'asset'})
    daily_cons['trade_date'] = daily_cons.index.map(lambda x: x[0])
    daily_cons = daily_cons.set_index(['asset', 'trade_date'])
    daily_factor = factor.unstack()
    shared_index = list(set(daily_factor.index) & set(daily_cons.index))
    factor_constitute = daily_factor.loc[shared_index].dropna()
    factor_constitute = factor_constitute.sort_index(level=1)
    return pd.DataFrame(factor_constitute)


if __name__ == '__main__':  # 因子值最好是pivot那种类型的，不是的话也行
    freq = 'D'
    target = 'us_stock'
    start_date = '20200101'
    end_date = '20250101'
    stock_path = r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\sp500_factor_result_gtja100'
    save_path = os.path.join(r'D:\大学文档\MAFM\IP\AtrDataCode\backtest_result\因子回测结果',target,freq)
    alpha_list = glob.glob(os.path.join(stock_path, '*.csv'))
    alpha_df_list = list(map(lambda x: pd.read_csv(x,index_col=0,parse_dates=True), alpha_list))
    col_list = list(map(lambda x:re.findall(r'\\(\w+).csv',x)[0],alpha_list))
    tasks = [(alpha_df_list[i],col_list[i],freq,target,start_date,end_date) for i in range(len(alpha_df_list))]
    with Pool(18) as pool:
        pool.starmap(single_factor_test, tasks)