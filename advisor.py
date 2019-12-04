#%%
from backtesting import *

class Advisor:
    def __init__(self, invest):
        # 1) perf_pack : (class)Invest
        self.portfolio = invest.portfolio
        self.invest_period = invest.invest_period
        self.perf = invest.perf
        self._tmp = None
        score_analysis()

    def score_analysis(self, benchmark='kospi', check_period=10, check_gap=0.1):
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

if __name__ == "__main__":
    print("\n--------------------------------------- data_loading.py test ---------------------------------------")
    print("advisor test...")


    print("Complete!!")
    print("--------------------------------------- No problem!!!!!!!!!! ---------------------------------------", end='\n\n')



#%%
