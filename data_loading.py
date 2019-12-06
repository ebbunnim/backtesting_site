'''
data_loading.py
    1) 백테스팅 및 어드바이저에 필요한 데이터를 규격화시켜준다
    2) dir_data 에서 pkl(혹은 hdf)파일을 읽어와 필요한 형태로 파싱한다.
    3) Return = (Class)pack
        a. pack.price_pack     : 가격 관련 데이터 팩
        b. pack.market_pack    : 마켓 관련 데이터 팩
        c. pack.liquidity_pack : 수급 관련 데이터 팩
    4) Usage Example
        a. pack.price_pack.price.value : 수정주가
        b. pack.price_pack.open.value  : 수정시가
        c. pack.market_pack.kosdaq     : 코스닥
    Cf. 백테스팅 및 다른 모듈에서 활용성(이름에 대한 접근) 및 반복문 사용을 위해 Enum Class를 상속받아 만들어짐
        따라서, pack.(target).name 및 pack.(target).value 로 접근해야하며 각각 이름과 데이터를 담고 있다.
'''

#%%
from _setting import setting
import pandas as pd
import numpy as np
import time
from enum import Enum

# ==================== Read the Data Directory ====================
_dir = str(setting['dir_data'])
    
    
# ====================== Data Pack Structure ======================
class PackInfo_DataGuide():
    market_open = None
    market_close = None

    class price_pack(Enum): # Price Related Pack
        directory = _dir+'price_pack.pkl'
        price = '수정주가(원)'
        price_open = '수정시가(원)'
        price_high = '수정고가(원)'
        price_low = '수정저가(원)'
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
        directory = _dir+'market_pack.pkl'
        kospi = ['코스피','종가지수(포인트)']
        kospi_open = ['코스피','시가지수(포인트)']
        kospi_high = ['코스피','고가지수(포인트)']
        kospi_low = ['코스피','저가지수(포인트)']
        kospi_trading_volume = ['코스피','거래대금(원)']
        kosdaq = ['코스닥','종가지수(포인트)']
        kosdaq_open = ['코스닥','시가지수(포인트)']
        kosdaq_high = ['코스닥','고가지수(포인트)']
        kosdaq_low = ['코스닥','저가지수(포인트)']
        kosdaq_trading_volume = ['코스닥','거래대금(원)']
        

    class liquidity_pack(Enum): # Liquidity Pack
        directory = _dir+'liquidity_pack.pkl'
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
        self._read() # Read data 
        self._unpack() # Unpack files
        # Market Open Data
        self.Pack.market_open = (self.Pack.market_pack.kospi.value).dropna().index
        self.Pack.market_close = pd.DatetimeIndex(np.setdiff1d((self.Pack.market_pack.kospi.value).index, self.Pack.market_open))
    
    def _read(self):
        print('File loading... ', end='')
        start = time.time()
        self.price_data = pd.read_pickle(self.Pack.price_pack.directory.value)
        self.market_data = pd.read_pickle(self.Pack.market_pack.directory.value)
        self.liquidity_data = pd.read_pickle(self.Pack.liquidity_pack.directory.value)
        print('complete!! %.5f sec.' %(time.time()-start))
              
    def _unpack(self):
        print('File unpacking... ', end=''); start = time.time()
        # MARKET DATA UNPACK        
        for component in self.Pack.market_pack:
            if component.name == 'directory': continue
            component._value_ = self.market_data[component.value[0]][component.value[1]]
            component._value_.name = component.name
        # PRICE RELATED DATA UNPACK
        for component in self.Pack.price_pack:
            if component.name == 'directory': continue
            component._value_ = self.price_data.xs(component.value, level=1, axis=1)
        # SUPPLY&DEMAND UNPACK
        for component in self.Pack.liquidity_pack:
            if component.name == 'directory': continue
            component._value_ = self.liquidity_data.xs(component.value, level=1, axis=1)
        print('complete!! %.5f sec.' %(time.time()-start))
        
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
    print("--------------------------------------------- Success! ---------------------------------------------", end='\n\n')


# %%
