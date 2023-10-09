import os
import argparse
import requests
import datetime

from dateutil.relativedelta import relativedelta

import pandas as pd
from bs4 import BeautifulSoup

from preprocess import Preprocess
from config.config import config_handler
from utils.notification import *


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

def get_holiday(hol_path, date, key, target_path):
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
        hol_df['hol_date'] = pd.to_datetime(hol_df['hol_date']).dt.date
        hol_df.to_csv(target_path, index=False)
    
    hol_df['hol_date'] = pd.to_datetime(hol_df['hol_date']).dt.date
    hol_df = hol_df.set_index('hol_date')
    print('getting holiday data is done!')
    return hol_df

class InferenceProcess(Preprocess):

    def __init__(self, args):
        super().__init__(args)
        # 휴일 데이터 불러오며 데이터 부족 시 추가 진행
        self.hol_df = get_holiday(hol_path = os.path.join(self.args.data_path, 'etc', self.args.holiday_name),
                                  date = self.args.today,
                                  key = self.args.config.get_value('PREPROCESS','holiday_api_key'),
                                  target_path = os.path.join(self.args.data_path, 'output', 'etc', self.args.holiday_name)
                                  )
        
    def logic(self, df):
        # total process
        df = self._transform_base_data(df)

        # pr data logic
        pr = self._make_pr_features(df)
        pr = self._get_pr_dataset(pr)

        # fr data logic
        fr = self._get_fr_dataset(df)

        # mr data logic
        mr = self._make_mr_features(df)
        mr = self._get_mr_dataset(mr)

        # save data
        fr_path = os.path.join(self.args.data_path, 'output','fr',self.args.data_name)
        mr_path = os.path.join(self.args.data_path, 'output','mr',self.args.data_name)
        pr_path = os.path.join(self.args.data_path, 'output','pr',self.args.data_name)
        
        fr.to_csv(fr_path, index=False, header=False)
        mr.to_csv(mr_path, index=False, header=False)
        pr.to_csv(pr_path, index=False, header=False)

        print('Save inference data is completed')

    def execution(self):
        df = pd.read_csv(os.path.join(self.args.data_path,self.args.today,'input',self.args.daily_data_name))
        
        self._integrate_data(df)
        self.logic(df)
        print('Inference preprocess is completed')
        
    def _integrate_data(self, df):
        # read integrated data
        intg_df = pd.read_csv(os.path.join(self.args.data_path,'integrate',self.args.data_name))
        # str -> date
        intg_df['std'] = pd.to_datetime(intg_df['std']).dt.date
        
        df['std'] = pd.to_datetime(df['std']).dt.date
        # data concat 
        intg_df = intg_df[~(intg_df['std']==df['std'].min())]
        intg_df = pd.concat([intg_df, df[df['std']==df['std'].min()]])
        # cut data
        cut_date = intg_df['std'].max() - relativedelta(months=30)
        
        intg_df = intg_df[intg_df['std']>=cut_date]
        # save data
        intg_df.to_csv(os.path.join(self.args.data_path,'output','integrate',self.args.data_name), index=False)

        print('Integrating data is completed')
        
if __name__=='__main__':
    try:
        parser = argparse.ArgumentParser(description='inference_preprocessing')

        parser.add_argument('--data_path', default='/opt/ml/processing')
        parser.add_argument('--data_name', default='pnr.csv')
        parser.add_argument('--daily_data_name', default='pnr.000')
        parser.add_argument('--holiday_name', default='holiday.csv')
        parser.add_argument('--today', type=str, default='20230725')
        parser.add_argument('--region', type=str, default="ap-northeast-2")
        parser.add_argument('--sns_arn', default='arn:aws:sns:ap-northeast-2:441848216151:awsdc-sns-dlk-dev-topic-ml-2023p01-lounge')
        parser.add_argument('--project_name', type=str)
        parser.add_argument('--pipeline_name', type=str)
        parser.add_argument('--env', type=str)
        
        args, _ = parser.parse_known_args()
        # get config file
        args.config = config_handler('preprocess_config.ini')
        prep = InferenceProcess(args)
        prep.execution()
    
    except Exception as e:

        publish_sns(
            region_name=args.region,
            sns_arn=args.sns_arn,
            project_name=args.project_name,
            pipeline_name=args.pipeline_name,
            error_type="Inference Preprocess Step 실행 중 에러",
            error_message=e,
            env=args.env,
        )

        raise Exception('step error')