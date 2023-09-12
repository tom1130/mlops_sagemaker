import os
import datetime
import requests
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def _get_holiday_info(year, month, key):
    '''
    year, month에서의 holiday 정보를 가져오기

        year : int
        month : int
        key : string

    '''
    url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getHoliDeInfo'
    params ={
        'serviceKey' : key,
        'solYear' : f'{year:4d}',
        'solMonth' : f'{month:02d}'
    }
    response = requests.get(url, params=params)
    
    soup = BeautifulSoup(response.content.decode('utf-8') , "xml")
    
    tmp_list = []
    for item in soup.items:
        if item.isHoliday.text == 'Y':
            hol_date = datetime.datetime.strptime(item.locdate.text, '%Y%m%d')
            hol_name = item.dateName.text
            tmp_list.append([hol_date, hol_name])
    df = pd.DataFrame(tmp_list, columns=['hol_date', 'hol_name'])
    return df

def get_holiday(hol_path, date, key):
    '''
    s3에 존재하는 holiday data 불러오기
    현재 존재하는 holiday data가 현재 달과 동일하거나 작다면, holiday 데이터를 openai로부터
    불러와 concat을 하고 hol_path에 저장

        hol_path : string
        date(today date YYYYmmdd) : string
        key : string
    '''
    # read holiday data
    hol_df = pd.read_csv(hol_path)
    # get max holiday date for comparing with today date
    hol_date_max = hol_df['hol_date'].max()
    
    date_year = datetime.datetime.strptime(date, '%Y%m%d').year
    date_month = datetime.datetime.strptime(date, '%Y%m%d').month
    hol_year = datetime.datetime.strptime(hol_date_max, '%Y-%m-%d').year
    hol_month = datetime.datetime.strptime(hol_date_max, '%Y-%m-%d').month

    if date_year==hol_year and date_month>=hol_month:
        print(f'add holiday data')
        if date_month==12:
            hol_new_df = _get_holiday_info(date_year+1, 1, key)
            print(f'year/month : {date_year+1}/1')
        else:
            hol_new_df = _get_holiday_info(date_year, date_month+1, key)
            print(f'year/month : {date_year}/{date_month+1}')
        hol_df = pd.concat([hol_df, hol_new_df])
        hol_df.to_csv(hol_path, index=False)
    
    hol_df['hol_date'] = pd.to_datetime(hol_df['hol_date']).dt.date
    hol_df = hol_df.set_index('hol_date')
    print('getting holiday data is done!')
    return hol_df

class Preprocess:

    def __init__(self, args):

        self.args = args
        self.hol_df = get_holiday(hol_path = os.path.join(self.args.data_path, 'etc', self.args.holiday_name),
                                  date = self.args.today,
                                  key = self.args.config.get_value('PREPROCESS','holiday_api_key')
                                  )
        self.time_group = {
            'BK' : { 'start_time' :  0 }, # 모든 시간대의 비행기 모두 BK 에 입장 가능
            'LN' : { 'start_time' : 11 }, # 11시 이후 비행기만 LN 대상에 포함 가능
            'DN' : { 'start_time' : 17 }, # 17시 이후 비행기만 LN 대상에 포함 가능
        }

    def logic(self):
        pass

    def execution(self):
        pass
        
    def _preprocess_label(self, df):
        df['std'] = pd.to_datetime(df['_date']).dt.date
        
        df['FR_LNG'] = df.loc[(df['lng_type'] == 'FR'), 'ke'].astype('int16')
        df['MR_LNG'] = df.loc[(df['lng_type'] == 'MR'), 'ke'].astype('int16')
        df['PR_LNG'] = df.loc[(df['lng_type'] == 'PR'), 'ke'].astype('int16')

        df_agg = df.groupby(['time_group', 'std']).agg({
            c : 'sum' for c in ['FR_LNG', 'MR_LNG', 'PR_LNG']
        }).astype('int16').reset_index().set_index('std')

        df_agg.index.name = 'std'
        df_agg = df_agg.rename(columns = {'time_group' : 'group'})
        return df_agg
        
    def _transform_base_data(self, df):
        '''
        1. na 값 치환
            - svc keyword 컬럼 na 값 치환
            - cbn_cls 컬럼 na 값 치환
        2. date 형식 지정
        3. 지역 구분
        '''
        # svc keyword 컬럼 na -> 0
        svc_columns = ['calf', 'calm', 'calp', 'hdcp', 'lngf', 'lngm', 'lngp', 'lngw', 'frdg', 'prdg', 'stfd']
        df[svc_columns] = df[svc_columns].fillna(0)
        
        # cbn_cls na -> bkg_cls로 치환
        bkg_cls_to_cbn_cls_dict = self.args.config.get_value('PREPROCESS','bkg_cls_to_cbn_cls_dict',dtype='dict')
        cbn_cls_mask = df['bkg_cls'].map(bkg_cls_to_cbn_cls_dict)
        df.loc[:, 'cbn_cls'] = df['cbn_cls'].apply(lambda x: np.NaN if x=='~' else x).fillna(cbn_cls_mask)

        # date 형식 지정
        df['std'] = pd.to_datetime(df['std']).dt.date

        # 지역구분 정리 : AME-EUR, CHN-JPN, SEA, KOR, ETC
        df['RGN_AMEEUR'] = df['arr_rgn'].isin(['AME', 'EUR'])
        df['RGN_CHNJPN'] = df['arr_rgn'].isin(['CHN', 'JPN'])
        df['RGN_KOR'] = (df['arr_rgn'] == 'KOR')
        df['RGN_SEA'] = (df['arr_rgn'] == 'SEA')
        df['RGN_ETC'] = df['arr_rgn'].isin(['OCN', 'MEA', 'CIS'])
        
        df['F'] = df.loc[(df['cbn_cls'] == 'F'), 'pax_count'].astype('int16')
        
        for cls in 'CY':
            for rgn in ['RGN_AMEEUR', 'RGN_SEA', 'RGN_CHNJPN', 'RGN_KOR', 'RGN_ETC']:
                col_name = rgn[4:] + '_' + cls
                df[col_name] = df.loc[((df['cbn_cls'] == cls) & df[rgn]), 'pax_count'].astype('int16')
        
        df['MM'] = (df['ffp_ke'] == 'MM')
        df['MP'] = (df['ffp_ke'] == 'MP')
        df['MC'] = (df['ffp_ke'] == 'MC')

        return df
    
    def _drop_useless_date(self, df):
        dates = self.args.config.get_value('PREPROCESS','dates',dtype='list')
        # change to date type
        dates = pd.to_datetime(dates).date
        # drop data
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
        agg_cols = ['F', 'lngf', 'calf', 'frdg']
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
                'calp', 'hdcp', 'lngp', 'prdg']

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
        df = df[df_cols+[f'{self.args.lounge_name}_LNG']]
        df = df.rename(columns={f'{self.args.lounge_name}_LNG' : 'target'})

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