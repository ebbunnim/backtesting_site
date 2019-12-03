#%%
from _setting import setting
import pandas as pd
import numpy as np
import time
from enum import Enum


# ==================== Read the Data Directory ====================
data_dir = str(setting['data_dir'])
    
    
# ====================== Data Pack Structure ======================
class PackInfo_DataGuide():
    market_open = None
    market_close = None
    class price_pack(Enum): # Price Related Pack
        directory = data_dir+'price_pack.pkl'
        price_open = '수정시가(원)'
        price_high = '수정고가(원)'
        price_low = '수정저가(원)'
        price = '수정주가(원)'
        return_1w = '수익률 (1주)(%)'
        return_1m = '수익률 (1개월)(%)'
        return_3m = '수익률 (3개월)(%)'
        return_6m = '수익률 (6개월)(%)'
        return_12m = '수익률 (12개월)(%)'
        eturn_ytd = '수익률 (YTD)(%)'
        vol_5d = '변동성 (5일)'
        vol_20d = '변동성 (20일)'
        vol_60d = '변동성 (60일)'
        vol_120d = '변동성 (120일)'
        vold_52w = '변동성 (52주)'
        
    class market_pack(Enum): # Market Pack
        directory = data_dir+'market_pack.pkl'
        kospi_open = '시가지수(포인트)'
        kodaq_open = '시가지수(포인트)'
        kospi_high = '고가지수(포인트)'
        kodaq_high = '고가지수(포인트)'
        kospi_low = '저가지수(포인트)'
        kodaq_low = '저가지수(포인트)'
        kospi = '종가지수(포인트)'
        kodaq = '종가지수(포인트)'
        kospi_trading_volume = '거래대금(원)' 
        kodaq_trading_volume = '거래대금(원)'   
        # market_open = None
        
    class liquidity_pack(Enum): # Supply and Demand Pack
        directory = data_dir+'liquidity_pack.pkl'
        inst_sell = '매도대금(기관계)(만원)'
        inst_buy = '매수대금(기관계)(만원)'
        inst = '순매수대금(기관계)(만원)'
        foreign_sell = '매도대금(외국인계)(만원)'
        foreign_buy = '매수대금(외국인계)(만원)'
        foreign = '순매수대금(외국인계)(만원)'
        individual_sell = '매도대금(개인)(만원)'
        individual_buy = '매수대금(개인)(만원)'
        individual = '순매수대금(개인)(만원)'
            
    class QP(Enum): # Quality Related Pack
        pass
    
    def __init__(self):
        pass
    

# ====================== Construct Data Pack ======================
class DataGuideData(PackInfo_DataGuide):    
    def __init__(self):
        # Pack Data - PRP, MP, SDP, QP
        # self.PackInfo = PackInfo_DataGuide() # Pack 구성 정보를 담을 Pack Information Set
        self.Pack = PackInfo_DataGuide() # Pack 구성 정보를 바탕으로 개별 데이터가 저장될 Data Set
        self.PRD = None # Parsing 대상이 되는 전체 Raw Data - Price Related
        self.MD = None # Parsing 대상이 되는 전체 Raw Data - Market
        self.SDD = None # Parsing 대상이 되는 전체 Raw Data - Supply & Demand 
        self.QD = None # Parsing 대상이 되는 전체 Raw Data - Quality
        # self.monthly_return = None
        # self.monthly_return_ie = None
        self._read() # Read data 
        self._unpack() # Unpack files
        # Market Open
        self.Pack.market_open = (self.Pack.market_pack.kospi.value).dropna().index
        self.Pack.market_close = pd.DatetimeIndex(np.setdiff1d((self.Pack.market_pack.kospi.value).index, self.Pack.market_open))
    
    def _read(self):
        print('File loading... ', end='')
        start = time.time()
        self.price_data = pd.read_pickle(self.Pack.price_pack.directory.value)
        self.market_data = pd.read_pickle(self.Pack.market_pack.directory.value)
        self.liquidity_data = pd.read_pickle(self.Pack.liquidity_pack.directory.value)
        print('complete!!', time.time()-start, 'sec')
              
    def _unpack(self):
        print('File unpacking... ', end='')
        start = time.time()
        # PRICE RELATED DATA UNPACK
        for i, component in enumerate(self.Pack.price_pack):
            if component.name == 'directory': continue
            component._value_ = self.price_data.xs(tuple(self.Pack.price_pack)[i].value, level=1, axis=1)

        # SUPPLY&DEMAND UNPACK
        for i, component in enumerate(self.Pack.liquidity_pack):
            if component.name == 'directory': continue
            component._value_ = self.liquidity_data.xs(tuple(self.Pack.liquidity_pack)[i].value, level=1, axis=1)
        
        # MARKET DATA UNPACK
        for i, component in enumerate(self.Pack.market_pack):
            if component.name == 'directory': continue
            mkt = '코스피' if component.name[:5] == 'kospi' else '코스닥'
            component._value_ = (self.market_data[mkt])[tuple(self.Pack.market_pack)[i].value]
        print('complete', time.time()-start, 'sec')
        
    # Data integrity check    
    def _updateData(self):
        pass

    
# ====================== Generate Data Pack =======================
_DT = DataGuideData()
pack = _DT.Pack


# ===================== __name__=="__main__" ======================
if __name__ == "__main__":
    print("\n--------------------------------------- data_loading.py test ---------------------------------------")
    print("data_loading test...")
    print(pack.market_open)
    print("Complete!!")
    print("--------------------------------------- No problem!!!!!!!!!! ---------------------------------------", end='\n\n')


# %%
