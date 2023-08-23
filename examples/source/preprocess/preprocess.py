import os
import sys
import datetime
import argparse
import requests
from typing import List, Union
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import unquote

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
    # APIKEY 관련 호출 방안 처리 필요 -> config 처리 예정
    HOLIDAY_API_KEY = 'tfL4n7XV90fA7+tbWFGqShE/JokqLQxd+0I89UfkMmTYxqXPR2nmWCeAr957kAtDh2U1BNq3fh2m3S0kc9fPUA=='
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
        # return hol_df

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

    try:
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
    except:
        print('Connection Error')
    #     hol_info = pd.read_csv('c:\\Users\\GIHO\\Desktop\\DS\\KE_lounge\\mlops_sagemaker\\examples\\data\\raw\\holiday.csv')
        hol_info = pd.read_csv('c://Users/고기호/Desktop/vscode/mlops/examples/data/raw/holiday.csv')
    hol_info['hol_date'] = pd.to_datetime(hol_info['hol_date']).dt.date
    hol_info = hol_info.set_index('hol_date')
    return hol_info

class Preprocess:

    def __init__(self, args):
        self.args = args
        self.hol_df = get_holiday_by_year(years = args.listYears)
        self.time_group = {
            'BK' : { 'start_time' :  0 }, # 모든 시간대의 비행기 모두 BK 에 입장 가능
            'LN' : { 'start_time' : 11 }, # 11시 이후 비행기만 LN 대상에 포함 가능
            'DN' : { 'start_time' : 17 }, # 17시 이후 비행기만 LN 대상에 포함 가능
        }

        self.svc_columns = ['calf', 'calm', 'calp', 'hdcp', 'lngf', 'lngm', 'lngp', 'lngw', 'frdg', 'prdg', 'stfd', 'sss']
        self.bkg_cls_to_cbn_cls_dict = {
            'P' : 'F',
            'F' : 'F',
            'A' : 'F',

            'J' : 'C',
            'C' : 'C',
            'D' : 'C',
            'I' : 'C',
            'R' : 'C',
            'Z' : 'C',
            'O' : 'C',

            'W' : 'Y',
            'Y' : 'Y',
            'B' : 'Y',
            'M' : 'Y',
            'S' : 'Y',
            'H' : 'Y',
            'E' : 'Y',
            'K' : 'Y',
            'L' : 'Y',
            'U' : 'Y',
            'Q' : 'Y',
            'G' : 'Y',
            'N' : 'Y',
            'T' : 'Y',
            'X' : 'Y',
            'V' : 'Y'
        }
    def logic(self):
        pass

    def execution(self):
        pass
    
    def _read_data(self):
        # 
        df = pd.read_csv(os.path.join(self.args.strDataPath, self.args.strDataName))
        return df
    
    def _read_label(self):
        label = pd.read_csv(os.path.join(self.args.strDataPath, self.args.strLabelName))
        
        label['std'] = pd.to_datetime(label['_date']).dt.date
        
        label['FR_LNG'] = label.loc[(label['lng_type'] == 'FR'), 'ke'].astype('int16')
        label['MR_LNG'] = label.loc[(label['lng_type'] == 'MR'), 'ke'].astype('int16')
        label['PR_LNG'] = label.loc[(label['lng_type'] == 'PR'), 'ke'].astype('int16')

        label_agg = label.groupby(['time_group', 'std']).agg({
            c : 'sum' for c in ['FR_LNG', 'MR_LNG', 'PR_LNG']
        }).astype('int16').reset_index().set_index('std')

        label_agg.index.name = 'std'
        label_agg = label_agg.rename(columns = {'time_group' : 'group'})
        return label_agg

    def _transform_base_data(self, df):
        '''
        1. na 값 치환
            - svc keyword 컬럼 na 값 치환
            - cbn_cls 컬럼 na 값 치환
        2. date 형식 지정
        3. 지역 구분
        '''
        # svc keyword 컬럼 na -> 0
        df[self.svc_columns] = df[self.svc_columns].fillna(0)
        
        # cbn_cls na -> bkg_cls로 치환
        cbn_cls_mask = df['bkg_cls'].map(self.bkg_cls_to_cbn_cls_dict)
        df.loc[:, 'cbn_cls'] = df['cbn_cls'].apply(lambda x: np.NaN if x=='~' else x).fillna(cbn_cls_mask)

        # date 형식 지정
        df['std'] = pd.to_datetime(df['std']).dt.date

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
        
        df['MM'] = (df['ffp_ke'] == 'MM')
        df['MP'] = (df['ffp_ke'] == 'MP')
        df['MC'] = (df['ffp_ke'] == 'MC')

        return df
    
    def _drop_useless_date(self, df):
        dates = [
            '20220513',
            '20221002',
            '20221003',
            '20221104',
            '20221105',
            '20230111',
            '20230112',
            '20230113',
            '20230114'
        ]
        # change to date type
        # dates = []??
        df = df[~df['std'].isin(dates)]

        return df

    def _make_mr_features(self, df):
        df['ELIP'] = (df['ffp_skyteam'] == 'ELIP')

        isJC = df['bkg_cls'].isin(['J', 'C'])

        isJCandAMEEUR = isJC & df['RGN_AMEEUR']
        df['JC_AMEEUR'] = df.loc[isJCandAMEEUR, 'pax_count']

        df['MM_PR'] = df.loc[df['MM'] & (df['cbn_cls'] == 'C'), 'pax_count'].astype('int16')
        df['MP_PR'] = df.loc[df['MP'] & (df['cbn_cls'] == 'C'), 'pax_count'].astype('int16')

        return df

    def _make_pr_features(self, df):
        df['EY_MM'] = df['MM'] & (df['cbn_cls'] == 'Y')
        df['EY_MP'] = df['MP'] & (df['cbn_cls'] == 'Y')
        df['EY_ELIP'] = (df['ffp_skyteam'] == 'ELIP') & (~df['MM']) & (~df['MP']) & (df['cbn_cls'] == 'Y')

        for new_col in ['AMEEUR_Y_MC', 'SEA_Y_MC', 'CHNJPN_Y_MC', 'KOR_Y_MC', 'ETC_Y_MC']:
            orgn_col = new_col[:-3]
            df[new_col] = df.loc[df[orgn_col].astype(bool) & df['MC'], 'pax_count']

        return df

    def _get_fr_dataset(self, df):
        # select fr columns
        agg_cols = ['AMEEUR_F', 'lngf', 'calf', 'frdg']
        not_agg_cols = ['staff_bkg', 'group']
        fr = pd.DataFrame(columns=agg_cols+not_agg_cols, dtype='int16')    

        for group, time_gt in self.time_group.items():
            # 임시 DataFrame 생성 : time group에 해당하는 것만 빼는 중
            tmp_df = df[
                df['std_hour'] >= time_gt['start_time']
            ]
            # tmp 중에서 확약 data만 추출
            confirmed_df = tmp_df[
                tmp_df['rsvn_code'] == 'HK'
            ]

            confirm_agg = confirmed_df.groupby('std').agg({
                c : 'sum' for c in agg_cols
            }).astype('int16')

            staff_agg = tmp_df[
                (tmp_df['rsvn_code'] == 'SA')
                & (tmp_df['cbn_cls'] == 'F')
            ].groupby('std').agg({
                'pax_count' : sum
            })

            staff_agg = staff_agg.rename(columns={'pax_count':'staff_bkg'})

            tmp_agg = confirm_agg.merge(staff_agg, on='std', how='left')
            tmp_agg = tmp_agg.fillna(0).astype('int16')
            tmp_agg['group'] = group

            fr = pd.concat([fr, tmp_agg])

        fr.index.name = 'std'
        fr['is_holiday'] = fr.index.isin(self.hol_df.index).astype('int16')
        fr = fr.reset_index()

        return fr
    
    def _get_mr_dataset(self, df):
        # select mr columns
        agg_cols = ['JC_AMEEUR', 'MM_PR', 'MP_PR', 'calm', 'lngm']
        not_agg_cols = ['group']
        mr = pd.DataFrame(columns=agg_cols+not_agg_cols, dtype='int16')

        for group, time_gt in self.time_group.items():
            tmp_df = df[df['std_hour'] >= time_gt['start_time']]
            
            confirmed_df = tmp_df[tmp_df['rsvn_code'] == 'HK']
            confirm_agg = confirmed_df.groupby('std').agg({
                c : 'sum' for c in agg_cols
            }).astype('int16')
            
            confirm_agg['group'] = group
            
            mr = pd.concat([mr, confirm_agg])

        mr.index.name = 'std'
        mr['is_holiday'] = mr.index.isin(self.hol_df.index).astype('int16')
        mr = mr.reset_index()

        return mr

    def _get_pr_dataset(self, df):
        agg_cols = ['AMEEUR_C', 'SEA_C', 'CHNJPN_C', 'KOR_C', 'ETC_C',
                'EY_MM', 'EY_MP', 'EY_ELIP',
                'AMEEUR_Y_MC', 'SEA_Y_MC', 'CHNJPN_Y_MC', 'KOR_Y_MC', 'ETC_Y_MC', 
                'AMEEUR_Y', 'SEA_Y', 'CHNJPN_Y', 'KOR_Y', 'ETC_Y', 
                'calp', 'hdcp', 'lngp', 'prdg', 'sss']

        not_agg_cols = ['stfd','group']

        pr = pd.DataFrame(columns=agg_cols+not_agg_cols, dtype='int16')

        for group, time_gt in self.time_group.items():
            tmp_df = df[
                df['std_hour'] >= time_gt['start_time']
            ]
            
            confirmed_df = tmp_df[tmp_df['rsvn_code'] == 'HK']
            confirm_agg = confirmed_df.groupby('std').agg({
                c : 'sum' for c in agg_cols
            }).astype('int16')
            
            staff_agg = tmp_df[
                tmp_df['cbn_cls'] != 'F'
            ].groupby('std').agg({
                'stfd' : 'sum'
            })
            
            tmp_agg = confirm_agg.merge(staff_agg, on='std', how='left')
            tmp_agg = tmp_agg.fillna(0).astype('int16')
            tmp_agg['group'] = group
            
            pr = pd.concat([pr, tmp_agg])

        pr.index.name = 'std'
        pr['is_holiday'] = pr.index.isin(self.hol_df.index).astype('int16')
        pr = pr.reset_index()

        return pr
    
    def _merge_data_label(self, df, label):
        df_cols = df.columns.tolist()

        df = df.merge(label, on=['std','group'], how='inner')
        df = df[df_cols+[f'{self.args.strLoungeName}_LNG']]
        df = df.rename(columns={f'{self.args.strLoungeName}_LNG' : 'target'})

        return df

    def _train_test_split(self, df):
        max_date = df['std'].max()
        valid_date = max_date - relativedelta(months=2)
        test_date = max_date - relativedelta(months=1)

        train = df[df['std']<valid_date]
        validation = df[(df['std']>=valid_date)&(df['std']<test_date)]
        test = df[df['std']>=test_date]

        return train, validation, test

if __name__=='__main__':
    pass