'''
backtesting.py
    1) 거래비용(중개 수수료 및 세금)과 실제 주문가능 수량을 고려한 정교한 백테스팅 모듈
    2) backtest(strategy, init_deposit, buy, sell, asset_name, fee_rate, tax_rate, test_from, test_to)
        a. strategy     : 투자 유니버스를 칼럼으로 리벨런스 날짜를 인덱스, 종목별 비중을 데이터로 갖는 데이터프레임(pickle file)
        b. init_deposit : 초기 투자금
        c. buy          : 매수 기준 가격
        d. sell         : 매도 기준 가격
        e. asset_name   : 자산 이름 지정
        f. fee_rate     : 기관 중개 수수료(매수 및 매도시 적용)
        g. tax_rate     : 세금(매도시 적용)
        h. test_from    : 백테스팅 기간 시작날짜(전략 파일 최초일과 동일하거나 이후여야 한다) ex. '2000-01-03'
        i. test_to      : 백테스팅 기간 종료날짜(전략 파일 최종일과 동일하거나 이전이여야 한다) ex. '2019-10-31'
    3) Return = (Class)Invest
        a. Basic Information
        b. Performance Information
    4) Usage Example
        a. from backtesting import *
    Cf. 백테스팅 및 다른 모듈에서 활용성(이름에 대한 접근) 및 반복문 사용을 위해 Enum Class를 상속받아 만들어짐
        따라서, pack.(target).name 및 pack.(target).value 로 접근해야하며 각각 이름과 데이터를 담고 있다.
'''

#%%
import os
import pandas as pd
import numpy as np
import time
import random
import ffn
import matplotlib.pyplot as plt
from data_loading import pack
from _setting import setting



# =================== Strategy Sample Loading ===================
dir_sample = setting['dir_strategy_sample']
strategy_sample = {}
for target in os.listdir(dir_sample):
    if target[-4:] == ".pkl":
        strategy_sample[target[:-4]] = pd.read_pickle(dir_sample+target)



# ==================== User Define Functions ====================
# Eliminate Unnecessary Parts of Large_df based on Small_df
def reduction(large_df, small_df):
    return large_df.loc[small_df.index,small_df.columns]

# Return emarket_packty DataFrame 
def plate(df):
    return pd.DataFrame(columns=df.columns, index=df.index)

# Return emarket_packty Series
def stick(df, stick_name=None):
    return pd.Series(index=df.index, name=stick_name)



# ======================= Investing Class =======================
class Performance:
    def __init__(self):
        self.value = None
        self.rets = None
        self.stats = None
        self.mdd = None

class Investing:
    def __init__(self, name):
        # 1) 자산 이름
        self.asset_name = name
        self.perf = Performance()
        self._tmp = None
        self.buy_at = 'open'
        self.sell_at = 'open'
        
    def activate(self, init_invest, dates, items):
        # 1) init_invest : 초기 투자금액
        # 2) period : 투자 기간내 유의한 날짜(자산 배분 변화일)
        # 3) items : 투자 자산군 내 개별 종목들
        ## Trading dates validation
        def _tradeVaildationTest(dates):
            if (np.intersect1d(dates, pack.market_close)).size != 0:
                print("ERROR! - The strategy includes transactions on a non-tradeable date.")
                return True
        # Validation Test
        if _tradeVaildationTest(dates):
            raise AssertionError
            return None
        # Activate         
        ind_init = list(pack.market_open).index(dates[0])
        ind_end = list(pack.market_open).index(dates[-1])
        self.invest_period = pack.market_open[ind_init:ind_end+1]
        self.rebal_dates = dates
        self.init_invest = init_invest 
        self.portfolio = pd.DataFrame(data=np.zeros((dates.size, items.size)), # 리밸런스시 보유할 종목 수
                                    index=dates, columns=items)
        self.trade = pd.DataFrame(data=np.zeros((dates.size, items.size)), # 주문 수
                                index=dates, columns=items)
        self.wallet = pd.DataFrame(data=np.zeros((dates.size, 3)), # 자산 가치 평가
                                index=dates, columns=[self.asset_name, 'cash', 'total'])
        self.wallet.cash = init_invest
        self.wallet.total = init_invest
        print("\t\t Activate success!!")
    
    def expansion(self, estimate_price):
        self.estimate_price = estimate_price

        # Generate Empty DataFrame for expansion
        plate_portfolio = pd.DataFrame(index=self.invest_period, columns=self.portfolio.columns)
        plate_wallet = pd.DataFrame(index=self.invest_period, columns=self.wallet.columns) 

        # Expansion Function for high speed Converting
        def _expansion_portfolio(row):
            if row.name in self.rebal_dates: # 리밸런스 날짜인 경우 - 대상 업데이트
                self._tmp = self.portfolio.loc[row.name]
            return self._tmp
        def _expansion_wallet(row):
            if row.name in self.rebal_dates: # 리밸런스 날짜인 경우 - 대상 업데이트
                self._tmp = self.wallet.loc[row.name]
            return self._tmp

        # Expansion
        self.portfolio = (plate_portfolio.T.apply(lambda row: _expansion_portfolio(row))).T
        self.wallet = (plate_wallet.T.apply(lambda row: _expansion_wallet(row))).T
        self.wallet[self.asset_name] = (self.portfolio * estimate_price).sum(axis=1)

    def performance(self):
        self._tmp = pd.concat([self.wallet.total, (pack.market_pack.kospi.value)[self.invest_period], (pack.market_pack.kosdaq.value)[self.invest_period]], axis=1)
        self.perf.value = self._tmp.rebase()
        self.perf.rets = self._tmp.to_returns()
        self.perf.stats = self._tmp.calc_stats()
        self.perf.mdd = self._tmp.to_drawdown_series()
    
    def show(self, benchmark='kospi'):
        pass
        # benchmark = {'kospi':pack.market_pack.kospi[self.invest_period], 'kosdaq':pack.market_pack.kosdaq[self.invest_period]}

        

# ========================= Backtesting =========================
def backtest(port_weight, deposit, buy='open', sell='open', asset_name='stock', fee_rate=0.00015, tax_rate=0.003, test_from=False, test_to=False, _massage=''):
    print('* Backtesting - Invest class generating')
    print(_massage, end='')
    # Input Arguments
    ## 1) Port_weight  : 전략에 따라 매 리밸런스기 보유해야할 종목 비중
    ## 2) deposit      : 최초 투자금액
    ## 3) buy / sell   : 매수가격 및 매도가격 기준(open/close)
    ## 4) asset_name   : 자산명
    ## 5) fee_rate     : 기관 중개 수수료
    ## 6) tax_rate     : 세금
    
    # port_weight을 직접 주어주지 않는경우(파일 이름을 주는 경우)
    if type(port_weight)!=pd.core.frame.DataFrame:
        asset_name = port_weight[:-4]
        print('\t',port_weight,'is external file!')
        print('\t Attempting to read', port_weight,'in strategy folder... ', end='')
        port_weight = pd.read_pickle(setting['dir_strategy']+port_weight)
        print(' Success!')

    # 백테스팅 범위 지정 - Out of Range에 대한 부분 차후 업데이트 필요
    if test_from:
        port_weight = port_weight.loc[test_from:]
    if test_to:
        port_weight = port_weight.loc[:test_to]

    # 고속 연산을 위한 내부 함수
    def _backtesting(row):
        t = row.name
        t_before = port_weight.index[list(port_weight.index).index(t)-1]
        if t!= port_weight.index[0]: 
            # 첫번째 리벨런스가 아닌경우
            ## 1) 총 자산 파악 : 보유하고있는 포트폴리오의 청산가치 + 보유현금
            ## 2) 종목별 배정금액 계산 : 총 자산 * 종목별 비중
            ## 3) 리밸런스 후 포트폴리오
            ## 4) 기존 포트폴리오에서 새로운 포트폴리오로 변경하기 위해 필요한 주문량 파악
            prev_portfolio = invest.portfolio.loc[t_before] # 보유하고있는 포트폴리오
            budget = (prev_portfolio*price_sell.loc[t]).sum()+invest.wallet.loc[t_before, 'cash'] # 현금 + 갖고있던 자산의 청산가치
            asset_alloc = row*budget # 자산당 배정 금액
            curr_portfolio = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0) # 새롭게 보유할 포트폴리오
            invest.trade.loc[t] = curr_portfolio - prev_portfolio # 종목별 주문량
        else: 
            # 최추 투자인 경우
            ## 1) 종목별 배정금액 계산 : 최초 투자금액 * 종목별 비중
            ## 2) 포트폴리오 
            ## 3) 포트폴리오 구성을 위한 주문량 파악
            asset_alloc = row * invest.wallet.loc[t].cash 
            curr_portfolio = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0)
            invest.trade.loc[t] = curr_portfolio
        # 주문량 파악 후
        ## 1) 매수 주문 
        ## 2) 매도 주문
        ## 3) 총 매수 규모
        ## 4) 총 매도 규모
        trade_buy = (invest.trade.loc[t])[invest.trade.loc[t]>0] # 종목별 매수주문 수
        trade_sell = -(invest.trade.loc[t])[invest.trade.loc[t]<0] # 종목별 매도주문 수
        buy_amt = (trade_buy*price_buy.loc[t]).sum() # 총 매수주문 규모
        sell_amt = (trade_sell*price_sell.loc[t]).sum() # 총 매도주문 규모
        # 현금 변화 반영 - SettingWithCopyWarning
        invest.wallet.loc[t,'cash'] = (invest.wallet.cash.loc[t_before]) + sell_amt - buy_amt
        invest.wallet.loc[t,'total'] = (curr_portfolio*price_sell.loc[t]).sum() + invest.wallet.loc[t, 'cash'] # 현금 + 갖고있던 자산의 청산가치    # 투자 시뮬레이션 객체 생성
        # 포트폴리오 업데이트
        invest.portfolio.loc[t] = curr_portfolio 

    # 투자 클래스 생성 및 활성화
    print("\n\t\t ------------",asset_name,"------------")
    invest = Investing(asset_name)
    invest.activate(deposit, port_weight.index, port_weight.columns)
    
    # 매매 설정에 따라 기준가격 설정(기관 중개 수수료 및 세금 반영)
    price_dict = {'open':pack.price_pack.price_open.value.loc[invest.invest_period], 'close':pack.price_pack.price_open.value.loc[invest.invest_period]}
    price_buy = price_dict[buy]*(1+fee_rate)
    price_sell = price_dict[sell]*(1-fee_rate-tax_rate)

    # 투자 시뮬레이션
    port_weight.T.apply(lambda row: _backtesting(row))
    print("\t\t Backtesting success!!")

    # 확장(리밸런스기 제외 평가금액 반영)
    invest.expansion(price_sell)
    print("\t\t Expansion success!!")

    # 투자성과 분석
    invest.performance()    
    print("\t\t Performance analysis success!!")
    print("\t\t --------------- Generate!! ---------------\n")

    return invest

'''
def backtest_daily(port_weight, deposit, buy='open', sell='open', asset_name='stock', fee_rate=0.00015, tax_rate=0.003):
    # Input Arguments
    ## 1) Port_weight  : 전략에 따라 매 리밸런스기 보유해야할 종목 비중
    ## 2) deposit      : 최초 투자금액
    ## 3) buy / sell   : 매수가격 및 매도가격 기준(open/close)
    ## 4) asset_name   : 자산명
    ## 5) fee_rate     : 기관 중개 수수료
    ## 6) tax_rate     : 세금
    
    # 고속 연산을 위한 내부 함수
    def _backtesting(row):
        t = row.name
        t_before = port_weight.index[list(port_weight.index).index(t)-1]
        
        if t!= port_weight.index[0]: 
            # 첫번째 리벨런스가 아닌경우
            ## 1) 총 자산 파악 : 보유하고있는 포트폴리오의 청산가치 + 보유현금
            ## 2) 종목별 배정금액 계산 : 총 자산 * 종목별 비중
            ## 3) 리밸런스 후 포트폴리오
            ## 4) 기존 포트폴리오에서 새로운 포트폴리오로 변경하기 위해 필요한 주문량 파악
            prev_portfolio = invest.portfolio.loc[t_before] # 보유하고있는 포트폴리오
            invest.wallet.total.loc[t] = (prev_portfolio*price_sell.loc[t]).sum()+invest.wallet.loc[t_before].cash # 현금 + 갖고있던 자산의 청산가치
            asset_alloc = row*invest.wallet.loc[t].total # 자산당 배정 금액
            curr_portfolio = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0) # 새롭게 보유할 포트폴리오
            invest.trade.loc[t] = curr_portfolio-prev_portfolio # 종목별 주문량
            invest.portfolio.loc[t] = curr_portfolio # 포트폴리오 업데이트  
        else: 
            # 최추 투자인 경우
            ## 1) 종목별 배정금액 계산 : 최초 투자금액 * 종목별 비중
            ## 2) 포트폴리오 
            ## 3) 포트폴리오 구성을 위한 주문량 파악
            asset_alloc = row * invest.wallet.loc[t].cash 
            invest.portfolio.loc[t] = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0)
            invest.trade.loc[t] = invest.portfolio.loc[t]
        # 주문량 파악 후
        ## 1) 매수 주문 
        ## 2) 매도 주문
        ## 3) 총 매수 규모
        ## 4) 총 매도 규모
        trade_buy = (invest.trade.loc[t])[invest.trade.loc[t]>0] # 종목별 매수주문 수
        trade_sell = -(invest.trade.loc[t])[invest.trade.loc[t]<0] # 종목별 매도주문 수
        buy_amt = (trade_buy*price_buy.loc[t]).sum() # 총 매수주문 규모
        sell_amt = (trade_sell*price_sell.loc[t]).sum() # 총 매도주문 규모
        # 현금 변화 반영
        invest.wallet.cash.loc[t] = invest.wallet.cash.loc[t_before] + sell_amt - buy_amt

        
    # 투자 시뮬레이션 객체 생성
    invest = Investing(asset_name)
    invest.activate(deposit, port_weight.index, port_weight.columns)
    
    # 매매 설정에 따라 기준가격 설정(기관 중개 수수료 및 세금 반영)
    price_buy = reduction(pack.price_pack.price_open.value, port_weight)*(1+fee_rate)
    price_sell = reduction(pack.price_pack.price.value, port_weight)*(1-fee_rate-tax_rate)
    
    # 투자 시뮬레이션
    port_weight.T.apply(lambda row: _backtesting(row))
    return invest
'''


# ======================= Sample Investment =======================
sample_num = 1
sample_ord = list(range(0, sample_num)); random.shuffle(sample_ord)

init_deposit = 100000000
invest_from = '2019-01-01'
invest_to = '2019-09-30'

invest_samples = [backtest(strategy_sample[list(strategy_sample.keys())[i]], init_deposit, asset_name=list(strategy_sample.keys())[i], \
    test_from=invest_from, test_to=invest_to, _massage='\t Sample investment generating\n\t\t Sample with cost') for i in sample_ord]
invest_samples_nc = [backtest(strategy_sample[list(strategy_sample.keys())[i]], init_deposit, asset_name=list(strategy_sample.keys())[i], \
    test_from=invest_from, test_to=invest_to, fee_rate=0., tax_rate=0., _massage='\t Sample investment generating\n\t\t Sample without cost') for i in sample_ord]


# ===================== __name__=="__main__" ======================
if __name__ == "__main__":
    pass
    # print("\n--------------------------------------- backtesting.py test ---------------------------------------")
    # print("\n--------------------------------------------- Success! ---------------------------------------------", end='\n\n')
    

#%%
