'''
advisor.py
    1) 
    2) 
    3) 
    4) Usage 
        a. (Class)Invest 를 인자로 받아 활성화시킨다 ; (var_name) = Advisor(Invest)

'''
#%%
from backtesting import *
from collections import OrderedDict as od
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# ==================== User Define Functions ====================
def purify_series(series, name):
    return pd.Series(np.array(series), name=name)



# ===================== Rebalance Analyzer ======================
'''
(Class)RebalanceAnalysis
    * Params
        1) (Class)invest       = 백테스팅 결과 도출된 Invest 클래스
        2) (str/bool)benchmark = 벤치마크 설정 
        3) (float)check_gap    = out-perform, under-perform 기준값
    * Components
        1) win_dates  = 벤치마크 대비 out-perform 한 리밸런스 가치의 리밸런스 날짜
        2) lose_dates = 벤치마크 대비 under-perform 한 리밸런스 가치의 리밸런스 날짜
        3) perf       = 리밸런스 성과 (리밸런스일/투자종료일/수익률/벤치마크대비 수익률/벤치마크수익률)
        4) perf_win   = Out-perform한 리밸런스의 성과
        5) perf_lose  = Under-perform한 리밸런스의 성과
        6) porf_win   = Out-perform한 리밸런스의 리밸런스 날짜별 포트폴리오
        7) porf_lose  = Under-perform한 리밸런스의 리밸런스 날짜별 포트폴리오          
'''
class RebalanceAnalyzer:
    def __init__(self, invest, benchmark, check_gap):
        self.invest = invest
        self.benchmark = benchmark
        self.check_gap = check_gap

        r_date = invest.rebal_dates # 리밸런스 날짜        
        r_porf = invest.portfolio.loc[r_date] # 리밸런스 포트폴리오
        e_price = invest.estimate_price.loc[r_date] # 리밸런스 시점 평가 가격

        # 리밸런스 성과(마지막 레코드는 의미없음)
        rebal_perf_item = (((r_porf*(e_price.shift(-1)).loc[r_date])/(r_porf*(e_price).loc[r_date])-1))
        rebal_perf = (((r_porf*(e_price.shift(-1)).loc[r_date]).sum(axis=1)/(r_porf*(e_price).loc[r_date]).sum(axis=1)-1))
        rebal_perf.fillna(0., inplace=True)

        # 벤치마크 성과(마지막 레코드는 의미없음)
        benchmark = (invest.perf.value[benchmark]).loc[r_date]
        rebal_perf_benchmark = (benchmark.shift(-1)/benchmark-1)

        # 벤치마크 대비 성과
        rebal_perf_rel = rebal_perf - rebal_perf_benchmark # 투자전략의 리벨런스기 벤치마크 대비 가치
        rebal_perf_item_rel = rebal_perf_item.sub(rebal_perf, axis=0) # 투자전략의 종목별 리밸런스기 벤치마크 대비 가치

        # 벤치마크 대비 성과기준 out-perform / under-perform
        self.win_dates = (rebal_perf_rel[rebal_perf_rel>check_gap].dropna()).index
        self.lose_dates = (rebal_perf_rel[rebal_perf_rel<-check_gap].dropna()).index


        # 리밸런스 테이블 요소
        comp = [purify_series(rebal_perf.index[:-1],'rebalance_date'), purify_series(rebal_perf.index[1:],'exit_date'), \
            purify_series(rebal_perf[:-1],'return'), purify_series(rebal_perf_rel[:-1],'relative_return'), purify_series(rebal_perf_benchmark[:-1],'benchmark')]        
        self.perf = pd.concat(comp, axis=1)
        self.perf.set_index(['rebalance_date'], inplace=True)

        
        # Out-perform / Under-perform
        self.perf_win = self.perf.loc[self.win_dates]
        self.perf_lose = self.perf.loc[self.lose_dates]

        self.test = rebal_perf_item
        self.test2 = rebal_perf_item_rel
        # 리밸런스 포트폴리오 종목
        def _rebal_portfolio(dates):
            res = []
            for d in dates:
                tmp = (pd.concat([pd.Series((rebal_perf_item.loc[d]).dropna(), name='return'), pd.Series(rebal_perf_item_rel.loc[d].dropna(), name='return_relative')], axis=1))
                tmp.sort_values(by='return')
                tmp.loc['total'] = np.array(self.perf[['return', 'relative_return']].loc[d])
                tmp.index = pd.MultiIndex.from_product([[d], list(tmp.index)])
                res.append(tmp)
            return pd.concat(res, axis=0)
            
        self.porf_win = _rebal_portfolio(self.win_dates)
        self.porf_lose = _rebal_portfolio(self.lose_dates)

        self.bench_corr = self.perf[['relative_return', 'benchmark']].corr()
        self.bench_corr_win = self.perf_win[['relative_return', 'benchmark']].corr()
        self.bench_corr_lose = self.perf_lose[['relative_return', 'benchmark']].corr()

    def show(self, save_fig=False):
        # Performance Showing
        fig = plt.figure(figsize=(15,20))
        ax1 = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)

        p1 = ax1.plot(self.perf[['relative_return','benchmark']])
        ax1.legend(p1,['relative_return','benchmark'], loc='upper right', shadow=True)
        ax1.axhline(y=self.check_gap, c='r', linestyle='--', linewidth=3)
        ax1.axhline(y=-self.check_gap, c='b', linestyle='--', linewidth=3)
        ax1.set_title("Relative Return of Rebalance Portfolio")
        
        # Winning Show
        p2 = ax2.plot(self.perf_win[['return', 'relative_return', 'benchmark']])
        ax2.legend(p2,['return', 'relative_return', 'benchmark'], loc='upper right', shadow=True)
        ax2.axhline(y=self.check_gap, c='r', linestyle='--', linewidth=3)
        ax2.axhline(y=-self.check_gap, c='b', linestyle='--', linewidth=3)
        ax2.set_title("Win Rebalance Portfolio")

        # Losing Show
        p3 = ax3.plot(self.perf_lose[['return', 'relative_return', 'benchmark']])
        ax3.legend(p3,['return', 'relative_return', 'benchmark'], loc='upper right', shadow=True)
        ax3.axhline(y=self.check_gap, c='r', linestyle='--', linewidth=3)
        ax3.axhline(y=-self.check_gap, c='b', linestyle='--', linewidth=3)
        ax3.set_title("Lose Rebalance Portfolio")

        if save_fig:
            plt.savefig('BA_rebal.png')
        plt.show()

        print("Correlation of Portfolio Performance and Benchmark", self.bench_corr, sep='\n', end='\n\n\n')
        print("Correlation of Win Portfolio Performance and Benchmark", self.bench_corr_win, sep='\n', end='\n\n\n')
        print("Correlation of Lose Portfolio Performance and Benchmark", self.bench_corr_lose, sep='\n', end='\n\n\n')

# ====================== Sector Analyzer ========================
class SectorAnalyzer:
    def __init__(self):
        pass



# ====================== Cap Analyzer ========================
class CapAnalyzer:
    def __init__(self):
        pass


# ===================== Analyzer Class =======================
class Analyzer:
    def __init__(self, invest):
        # 1) perf_pack : (class)Invest
        self.invest = invest

    def activate(self, benchmark='kospi', check_gap=0.1):
        self.rebal = RebalanceAnalyzer(self.invest, benchmark, check_gap)
        self.sector = SectorAnalyzer()
        self.cap = CapAnalyzer()



# ========================= Samples ===========================


if __name__ == "__main__":
    print("\n----------------------------- backtesting_analyzer.py test -----------------------------")

    adv_nc = Analyzer(backtest('QVM_SD_6w121.pkl', 100000000, fee_rate=0., tax_rate=0.))
    adv_nc.activate()

    print("Complete!!")
    print("--------------------------------------- Success! ---------------------------------------", end='\n\n')




# %%


