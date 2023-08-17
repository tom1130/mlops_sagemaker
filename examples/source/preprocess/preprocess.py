import os
import sys
import datetime
import argparse
import requests
from typing import List, Union

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import unquote

def arg_parse():
    pass


def _get_holiday_info(year:int, month:int, hol_df:pd.DataFrame) -> pd.DataFrame:
    '''
    한국천문연구원에서 휴일 정보를 호출한 후, 받아온 정보를 정제하는 함수

    Parameters
    ---------------
    - year :
        - 휴일 정보 가져오려는 연도 
    - month : 
        - 휴일 정보 가져오려는 달
    - hol_df :
        - 휴일 정보를 저장할 dataframe
    
    Return
    ------------
    pd.DataFrame

    '''
    # APIKEY 관련 호출 방안 처리 필요
    HOLIDAY_API_KEY = unquote(os.environ['HOLIDAY_API_KEY'])
    url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getHoliDeInfo'
    params ={
        'serviceKey' : HOLIDAY_API_KEY,
        'solYear' : f'{year:4d}',
        'solMonth' : f'{month:02d}'
    }
    response = requests.get(url, params=params)
    
    # bs4 작업
    soup = BeautifulSoup(response.content.decode('utf-8') , "xml")
    # try:
    for item in soup.items:
        tmp_list = []
        if item.isHoliday.text == 'Y':
            hol_date = datetime.datetime.strptime(item.locdate.text, '%Y%m%d')
            hol_name = item.dateName.text
            tmp_list.append({'hol_date' : hol_date, 'hol_name': hol_name})
            
        tmp_df = pd.DataFrame(tmp_list)
        hol_df = pd.concat([hol_df, tmp_df])
    # except:
    #     return hol_df

    return hol_df

def get_holiday_by_year(year:Union[int, None] = None, years:Union[List[int], None] = None) -> pd.DataFrame:
    '''
    특정 연도의 휴일을 pd.DataFrame으로 반환하는 함수
    
    year와 years 중 하나만 입력해야 하며, 모두 입력하거나 모두 입력하니 않는 경우 ValueError 발생

    Parameters
    ----------
    - year : int
        - 휴일 정보를 가져오려는 연도
        - e.g. 2023
    
    - years : List[int]
        - 휴일 정보를 가져오려는 연도
        - e.g. [2023, 2024]

    Returns
    ----------
    pd.DataFrame

    '''
    if (year is None) & (years is None):
        raise ValueError

    if isinstance(year, int) & isinstance(years, list):
        raise ValueError
    
    hol_info = pd.DataFrame(columns=['hol_date', 'hol_name'])
    
    # year 지정시
    if isinstance(year, int):
        for month in range(1, 13):
            hol_info = _get_holiday_info(year, month, hol_info)
        hol_info = hol_info.set_index('hol_date')
        return hol_info

    # years 지정시
    for y in years:
        for month in range(1, 13):
            hol_info = _get_holiday_info(y, month, hol_info)

    hol_info = hol_info.set_index('hol_date')
    return hol_info

class preprocess:

    def __init__(self):
        pass

    def logic(self):
        pass

    def execution(self):
        pass
    
    def _read_data(self):
        df = pd.read_csv('args')
        return df

    def _(self, df):
        '''
        1. na 값 치환
            - svc keyword 컬럼 na 값 치환
            - cbn_cls 컬럼 na 값 치환
        2. date 형식 지정
        3. 지역 구분
        '''
        # svc keyword 컬럼 na -> 0
        df['args'] = df['args'].fillna(0)
        
        # cbn_cls na -> bkg_cls로 치환
        cbn_cls_mask = df['bkg_cls'].map('args')
        df.loc[:, 'cbn_cls'] = df['cbn_cls'].apply(lambda x: np.NaN if x=='~' else x).fillna(cbn_cls_mask)

        # 지역구분 정리 : AME-EUR, CHN-JPN, SEA, KOR, ETC
        df['RGN_AMEEUR'] = df['arr_rgn'].isin(['AME', 'EUR'])
        df['RGN_CHNJPN'] = df['arr_rgn'].isin(['CHN', 'JPN'])
        df['RGN_KOR'] = (df['arr_rgn'] == 'KOR')
        df['RGN_SEA'] = (df['arr_rgn'] == 'SEA')
        df['RGN_ETC'] = df['arr_rgn'].isin(['OCN', 'MEA', 'CIS'])

        for cls in 'FCY':
            for rgn in ['RGN_AMEEUR', 'RGN_SEA', 'RGN_CHNJPN', 'RGN_KOR', 'RGN_ETC']:
                col_name = rgn[4:] + '_' + cls
                df[col_name] = df.loc[((df['cbn_cls'] == cls) & df[rgn]), 'pax_count'].astype('int16')

        return df
    
    def _drop_useless_date(self, df):
        pass

    def _make_mr_features(self, df):
        pass

    def _make_pr_features(self, df):
        pass

    def _get_fr_dataset(self, df):
        pass

    def _get_mr_dataset(self, df):
        pass

    def _get_pr_dataset(self, df):
        pass

if __name__=='__main__':
    pass