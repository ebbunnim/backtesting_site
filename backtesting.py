#%%
from data_loading import pack
from _setting import setting
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

sample_dir = setting['strategy_sample_dir']
strategy_sample = {}
for target in os.listdir(sample_dir):
    if target[-4:] == ".pkl":
        strategy_sample[target[:-4]] = pd.read_pickle(sample_dir+target)

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
class Investing:
    def __init__(self, name):
        # 1) 자산 이름
        self.asset_name = name
        
    def activate(self, init_invest, dates, items):
        # 1) init_invest : 초기 투자금액
        # 2) period : 투자 기간내 유의한 날짜(자산 배분 변화일)
        # 3) items : 투자 자산군 내 개별 종목들
        self.init_invest = init_invest # 초기 투자값
        self.portfolio = pd.DataFrame(data=np.zeros((dates.size, items.size)), # 리밸런스시 보유할 종목 수
                                      index=dates, columns=items)
        self.trade = pd.DataFrame(data=np.zeros((dates.size, items.size)), # 주문 수
                                  index=dates, columns=items)
        self.wallet = pd.DataFrame(data=np.zeros((dates.size, 3)), # 자산 가치 평가
                                  index=dates, columns=[self.asset_name, 'cash', 'total'])
        self.wallet.cash = init_invest
        self.wallet.total = init_invest
        
        
        
# ========================= Backtesting =========================
def backtest(port_weight, deposit, buy='open', sell='open', asset_name='stock', fee_rate=0.00015, tax_rate=0.003):
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
    price_dict = {'open':reduction(pack.price_pack.price_open.value, port_weight), 'close':reduction(pack.price_pack.price.value, port_weight)}
    price_buy = price_dict[buy]*(1+fee_rate)
    price_sell = price_dict[sell]*(1-fee_rate-tax_rate)
    
    # 투자 시뮬레이션
    port_weight.T.apply(lambda row: _backtesting(row))
    return invest


# ===================== __name__=="__main__" ======================
if __name__ == "__main__":
    print("\n--------------------------------------- backtesting.py test ---------------------------------------")
    print("Elaborate Backtesting Test...")
    invest_test = backtest(strategy_sample['daily_strategy01'], 100000000)
    invest_test.wallet.total.plot()
    print("Complete!!")
    print("--------------------------------------- No problem!!!!!!!!! ---------------------------------------", end='\n\n')
    

# %%
