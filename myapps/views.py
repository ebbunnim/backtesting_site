from django.shortcuts import render
import os
import pandas as pd
import numpy as np
import time
import random
import ffn
import matplotlib.pyplot as plt
from data_loading import pack
from _setting import setting

from io import BytesIO
import base64
from django.http import HttpResponse
from backtesting_analyzer import *


def index(request):
    
    dir_sample = setting['dir_strategy_sample']
    strategy_sample = {}
    for target in os.listdir(dir_sample):
        # print(target)

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
    def backtest(port_weight, deposit, buy='open', sell='open', asset_name='stock', fee_rate=0.00015, tax_rate=0.003, test_from=False, test_to=False):
        # Input Arguments
        ## 1) Port_weight  : 전략에 따라 매 리밸런스기 보유해야할 종목 비중
        ## 2) deposit      : 최초 투자금액
        ## 3) buy / sell   : 매수가격 및 매도가격 기준(open/close)
        ## 4) asset_name   : 자산명
        ## 5) fee_rate     : 기관 중개 수수료
        ## 6) tax_rate     : 세금
        
        # port_weight을 직접 주어주지 않는경우(파일 이름을 주는 경우)

        # input request >> change default
        if request.method == "POST":
            # port_weight = request.POST['strategy']
            # deposit = float(request.POST['deposit'])
            buy = str(request.POST['buy'])
            sell = str(request.POST['sell'])
            # asset_name = str(request.POST['asset_name'])
            fee_rate = float(request.POST['fee_rate'])
            tax_rate = float(request.POST['tax_rate'])

        # -------exit--------

        if type(port_weight)==str:
            print('Read strategy file from strategy folder... ', end='')
            port_weight = pd.read_pickle(setting['dir_strategy_sample']+port_weight)
            print(' Complete!')

        if test_from:
            port_weight = port_weight.loc[test_from:]
        if test_to:
            port_weight = port_weight.loc[:test_to]

        # 고속 연산을 위한 내부 함수
        def _backtesting(row):
            t = row.name
            t_before = port_weight.index[list(port_weight.index).index(t)-1]
            if t!= port_weight.index[0]: 

                prev_portfolio = invest.portfolio.loc[t_before] # 보유하고있는 포트폴리오
                budget = (prev_portfolio*price_sell.loc[t]).sum()+invest.wallet.loc[t_before, 'cash'] # 현금 + 갖고있던 자산의 청산가치
                asset_alloc = row*budget # 자산당 배정 금액
                curr_portfolio = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0) # 새롭게 보유할 포트폴리오
                invest.trade.loc[t] = curr_portfolio - prev_portfolio # 종목별 주문량
            else: 

                asset_alloc = row * invest.wallet.loc[t].cash 
                curr_portfolio = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0)
                invest.trade.loc[t] = curr_portfolio

            trade_buy = (invest.trade.loc[t])[invest.trade.loc[t]>0] # 종목별 매수주문 수
            trade_sell = -(invest.trade.loc[t])[invest.trade.loc[t]<0] # 종목별 매도주문 수
            buy_amt = (trade_buy*price_buy.loc[t]).sum() # 총 매수주문 규모
            sell_amt = (trade_sell*price_sell.loc[t]).sum() # 총 매도주문 규모
            invest.wallet.loc[t,'cash'] = (invest.wallet.cash.loc[t_before]) + sell_amt - buy_amt
            invest.wallet.loc[t,'total'] = (curr_portfolio*price_sell.loc[t]).sum() + invest.wallet.loc[t, 'cash'] # 현금 + 갖고있던 자산의 청산가치    # 투자 시뮬레이션 객체 생성
            invest.portfolio.loc[t] = curr_portfolio 

        print("\n\t\t ------------",asset_name,"------------")
        invest = Investing(asset_name)
        invest.activate(deposit, port_weight.index, port_weight.columns)
        
        price_dict = {'open':pack.price_pack.price_open.value.loc[invest.invest_period], 'close':pack.price_pack.price_open.value.loc[invest.invest_period]}
        price_buy = price_dict[buy]*(1+fee_rate)
        price_sell = price_dict[sell]*(1-fee_rate-tax_rate)

        port_weight.T.apply(lambda row: _backtesting(row))
        print("\t\t Backtesting success!!")

        invest.expansion(price_sell)
        print("\t\t Expansion success!!")

        invest.performance()    
        print("\t\t Performance analysis success!!")
        print("\t\t --------------- Generate!! ---------------")

        return invest




    # ======================= Sample Investment =======================
    '''
    sample_num = 1
    sample_ord = list(range(0, sample_num)); random.shuffle(sample_ord)

    init_deposit = 100000000
    invest_from = '2019-01-01'
    invest_to = '2019-09-30'

    invest_samples = [backtest(strategy_sample[list(strategy_sample.keys())[i]], init_deposit, asset_name=list(strategy_sample.keys())[i], \
        test_from=invest_from, test_to=invest_to) for i in sample_ord]
    invest_samples_nc = [backtest(strategy_sample[list(strategy_sample.keys())[i]], init_deposit, asset_name=list(strategy_sample.keys())[i], \
        test_from=invest_from, test_to=invest_to, fee_rate=0., tax_rate=0.) for i in sample_ord]
    '''


    if request.method == "POST":
    
        # retry
        # port_weight = request.POST['strategy']
        # deposit = float(request.POST['deposit'])
        buy = str(request.POST['buy'])
        sell = str(request.POST['sell'])
        fee_rate = float(request.POST['fee_rate'])
        tax_rate = float(request.POST['tax_rate'])

        
        # three plot
        inv = backtest(strategy_sample[request.POST['strategy']], float(request.POST['deposit']))
        # inv.perf.value.plot()
        # inv.perf.mdd.plot()
        # inv.perf.stats.display() / inv.perf.stats.stats -> to dataframe

        buf = BytesIO()
        inv.perf.value.plot().cla()
        inv.perf.value.plot().get_figure().savefig(buf, format='png', dpi=300)
        image_base64_1 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()

        buf = BytesIO()
        inv.perf.mdd.plot().cla()
        inv.perf.mdd.plot().get_figure().savefig(buf, format='png', dpi=300)
        image_base64_2 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()

        # stats dataframe 가능한지 보기
        res = inv.perf.stats.stats.to_html()

        context = {
            'image_base64_1' : image_base64_1,
            'image_base64_2' : image_base64_2,
            'res': res,


        }
    else:
        context = {
            '1' : 1
        }

    return render(request, 'myapps/index.html', context)

def strategies(request):
    dir_sample = setting['dir_strategy_sample']
    strategy_list = []
    for target in os.listdir(dir_sample):
        # print(target)
        strategy_list.append(target)
    
    context = {
        'strategy_list': strategy_list, 
    }
    return render(request, 'myapps/strategies.html', context)

def analyze(request):
    return render(request, 'myapps/analyze.html')


def improve(request):
    return render(request, 'myapps/blank.html')