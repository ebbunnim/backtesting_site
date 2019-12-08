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
import pandas as pd
import numpy as np

def purify_series(series, name):
    return pd.Series(np.array(series), name=name)

class RebalanceAnalysis:
    def __init__(self, invest, benchmark, check_gap):
        '''
        * Params
            1) (Class)invest       = 백테스팅 결과 도출된 Invest 클래스
            2) (str/bool)benchmark = 벤치마크 설정 
            3) (float)check_gap    = 이상치 기준 

        * Components
            1) rebal_perf_item      : 리밸런스기 종목별 투자 성과
            2) rebal_perf           : 리밸런스의 투자 성과
            3) rebal_perf_benchmark : 벤치마크에 투자한 경우 리밸런스기 가치
            4) rebal_value          : 리밸런스 가치
        '''
        r_date = invest.rebal_dates # 리밸런스 날짜        
        r_porf = invest.portfolio.loc[r_date] # 리밸런스 포트폴리오
        e_price = invest.estimate_price.loc[r_date] # 리밸런스 시점 평가 가격

        # 리밸런스 성과(마지막 레코드는 의미없음)
        rebal_perf_item = ((r_porf*(e_price.shift(-1)).loc[r_date])/(r_porf*(e_price).loc[r_date])-1)
        rebal_perf = ((r_porf*(e_price.shift(-1)).loc[r_date]).sum(axis=1)/(r_porf*(e_price).loc[r_date]).sum(axis=1)-1)
        
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
        self.performance = pd.concat(comp, axis=1, ignore_index=True)
        self.win = self.performance.loc[self.win_dates]
        self.lose = self.performance.loc[self.lose_dates]

        # 리밸런스 포트폴리오 종목
        def _rebal_portfolio(dates):
            rebal_portfolio = {}
            for date in dates:
                rebal_portfolio[date] = (pd.concat([(rebal_perf_item.loc[date]).dropna(), (rebal_perf_item_rel.loc[date]).dropna()], axis=1, names=['return', 'relative_return'], ignore_index=True)).T
            return rebal_portfolio
        self.win_portfolio = _rebal_portfolio(self.win_dates)
        self.lose_portfolio = _rebal_portfolio(self.lose_dates)


class Advisor:
    def __init__(self, invest):
        # 1) perf_pack : (class)Invest
        self.invest = invest

    def activate(self, benchmark='kospi', check_gap=0.1):
        self.rebal = RebalanceAnalysis(self.invest, benchmark, check_gap)
        # self.sector = SectorAnalysis(invest, )

    def update(self):
        pass


# Sample
adv_nc = Advisor(invest_samples_nc[0])
adv_c = Advisor(invest_samples[0])
adv_nc.activate()
adv_c.activate()
r = adv_nc.rebal

if __name__ == "__main__":
    print("\n--------------------------------------- data_loading.py test ---------------------------------------")
    print("advisor test...")


    print("Complete!!")
    print("--------------------------------------- No problem!!!!!!!!!! ---------------------------------------", end='\n\n')



#%%











    '''
    def score_analysis(self, benchmark='kospi', check_period=10, check_gap=0.05):
        # 1) benchmark    : 비교할 벤치마크(kospi, kosdaq)
        # 2) check_period : 기간 수익률
        # 3) check_gap    : 이상치 기준

        ## 1. invest 단위 기간 수익률로 변환
        self._tmp = (self.perf.value/self.perf.value.shift(check_period)-1).dropna(how='all', axis=0)
        
        ## 2. 기간 수익률로 벤치마크대비 성과 및 out perform/under perform 계산
        self.score = (self._tmp.total-self._tmp[benchmark])
        self.score_win = self.score > check_gap
        self.score_lose = self.score < -check_gap
        self.win_dates = (self.score_win.replace(False, np.nan)).dropna().index
        self.lose_dates = (self.score_lose.replace(False, np.nan)).dropna().index
    
        ## 3. out perform/under perform 별 성과 분석
        self.lose_portfolio = None
        self.lose_rets_item = None
        self.lose_rets = None

    def rebal_analysis(self, benchmark='kospi', check_gap=0.1):
    '''

    '''
        self.win_ret = 

        # out/under시 
        self.win_ret = self.pref
        self.lose_ret
        self.win_portfolio
        self.lose_portfolio
        self.win_

        # 
        self.win_table = 

    def size_analysis(self):
        pass

    def industry_analysis(self):
        pass

    def value_analysis(self):
        pass
    '''
