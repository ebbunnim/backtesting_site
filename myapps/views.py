from django.shortcuts import render
from data_loading import pack
from _setting import setting
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from django.http import HttpResponse


# Create your views here.
def index(request):


    sample_dir = setting['strategy_sample_dir']
    strategy_sample = {}
    for target in os.listdir(sample_dir):
        if target[-4:] == ".pkl":
            strategy_sample[target[:-4]] = pd.read_pickle(sample_dir+target)

    def reduction(large_df, small_df):
        return large_df.loc[small_df.index,small_df.columns]
    def plate(df):
        return pd.DataFrame(columns=df.columns, index=df.index)
    def stick(df, stick_name=None):
        return pd.Series(index=df.index, name=stick_name)


    class Investing:
        def __init__(self, name):
            self.asset_name = name
            
        def activate(self, init_invest, dates, items):
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
        
        def _backtesting(row):
            t = row.name
            t_before = port_weight.index[list(port_weight.index).index(t)-1]
            
            if t!= port_weight.index[0]: 
                prev_portfolio = invest.portfolio.loc[t_before] # 보유하고있는 포트폴리오
                invest.wallet.total.loc[t] = (prev_portfolio*price_sell.loc[t]).sum()+invest.wallet.loc[t_before].cash # 현금 + 갖고있던 자산의 청산가치
                asset_alloc = row*invest.wallet.loc[t].total # 자산당 배정 금액
                curr_portfolio = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0) # 새롭게 보유할 포트폴리오
                invest.trade.loc[t] = curr_portfolio-prev_portfolio # 종목별 주문량
                invest.portfolio.loc[t] = curr_portfolio # 포트폴리오 업데이트  
            else: 
                asset_alloc = row * invest.wallet.loc[t].cash 
                invest.portfolio.loc[t] = (asset_alloc//(price_buy.loc[t])).replace(np.nan, 0)
                invest.trade.loc[t] = invest.portfolio.loc[t]
            trade_buy = (invest.trade.loc[t])[invest.trade.loc[t]>0] # 종목별 매수주문 수
            trade_sell = -(invest.trade.loc[t])[invest.trade.loc[t]<0] # 종목별 매도주문 수
            buy_amt = (trade_buy*price_buy.loc[t]).sum() # 총 매수주문 규모
            sell_amt = (trade_sell*price_sell.loc[t]).sum() # 총 매도주문 규모
            invest.wallet.cash.loc[t] = invest.wallet.cash.loc[t_before] + sell_amt - buy_amt

        invest = Investing(asset_name)
        invest.activate(deposit, port_weight.index, port_weight.columns)
        
        price_dict = {'open':reduction(pack.price_pack.price_open.value, port_weight), 'close':reduction(pack.price_pack.price.value, port_weight)}
        price_buy = price_dict[buy]*(1+fee_rate)
        price_sell = price_dict[sell]*(1-fee_rate-tax_rate)
        
        port_weight.T.apply(lambda row: _backtesting(row))
        return invest

    def pltToSvg(plt):
        buf = io.BytesIO()
        fig = plt.get_figure()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        s = buf.getvalue()
        buf.close()
        return s

    def get_svg(request, myplt):
        svg = pltToSvg(myplt) # convert plot to SVG
        myplt.cla() # clean up plt so it can be re-used
        response = HttpResponse(svg, content_type='image/svg+xml')
        return response


    if request.method == "POST":
        print('====================if POST========================')
        # print(request.POST['strategy_field'])
        invest_test = backtest(strategy_sample[request.POST['strategy_field']], 100000000)
    # invest_test.wallet.total.plot() 

        
        buf = BytesIO()
        invest_test.wallet.total.plot().cla()
        invest_test.wallet.total.plot().get_figure().savefig(buf, format='png', dpi=300)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()

        context = {
            'image_base64' : image_base64,
        }
    else:
        context = {
            '1' : 1
        }


    return render(request, 'myapps/index.html', context)

