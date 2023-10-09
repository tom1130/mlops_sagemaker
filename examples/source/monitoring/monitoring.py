import os
import argparse
import json
import boto3
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from utils.notification import *

class Monitoring:
    
    def __init__(self, args):
        self.args = args

        print(self.args.now)
        self.now = datetime.strftime(datetime.strptime(self.args.now, '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=9), '%H%M%S')
        self.today = datetime.strftime(datetime.strptime(self.args.now, '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=9), '%Y%m%d')
        print(self.args.now)
        print(self.today)
        # sagemaker, sns client 설정
        self.client = boto3.client('sagemaker', region_name=self.args.region_name)
        self.sns_client = boto3.client('sns', region_name=self.args.region_name)

    def logic(self):
        # prediction과 label merge
        self.df = self.df.merge(self.label, on=['date','time_group','lounge_type'], how='inner')

        # fr r2, mae result
        fr = self.df[self.df['lounge_type']=='FR']
        fr_r2 = r2_score(fr['pred'], fr['label'])
        fr_mae = mean_absolute_error(fr['pred'], fr['label'])
        # mr r2, mae result
        mr = self.df[self.df['lounge_type']=='MR']
        mr_r2 = r2_score(mr['pred'], mr['label'])
        mr_mae = mean_absolute_error(mr['pred'], mr['label'])
        # pr r2, mae result
        pr = self.df[self.df['lounge_type']=='PR']
        pr_r2 = r2_score(pr['pred'], pr['label'])
        pr_mae = mean_absolute_error(pr['pred'], pr['label'])
        
        # make dataFrame
        report = pd.DataFrame([[self.today, fr_r2, fr_mae, mr_r2, mr_mae, pr_r2, pr_mae]], columns=['date','FR_r2','FR_mae','MR_r2','MR_mae','PR_r2','PR_mae'])
        print(f'r2, mae result : {report}')
        # read exist output
        monitoring_results = pd.read_csv(os.path.join(self.args.data_path, 'input', 'monitoring', self.args.output_name))
        # concat
        monitoring_result = pd.concat([monitoring_results, report])
        # save monitoring data
        output_path = os.path.join(self.args.data_path, 'output', 'monitoring', self.args.output_name)
        monitoring_result.to_csv(output_path, index=False)
        print('Saving monitoring result is completed')

        # fr training pipeline execute if needed
        ## 3개의 변수 변경, now(현재 시간), today(현재 날짜), model_approval_status(모델 등록시, register option)
        if fr_mae > self.args.fr_mae_threshold:
            response = self.client.start_pipeline_execution(
                PipelineName=self.args.fr_pipeline_name,
                PipelineParameters=[
                    {
                        'Name' : 'now',
                        'Value' : self.now
                    },
                    {
                        'Name' : 'today',
                        'Value' : self.today
                    },
                    {
                        'Name' : 'model_approval_status',
                        'Value' : 'PendingManualApproval'
                    }
                ]
            )
            print(response)
            
        # mr training pipeline execute if needed
        if mr_r2 < self.args.mr_r2_threshold or mr_mae > self.args.mr_mae_threshold:
            response = self.client.start_pipeline_execution(
                PipelineName=self.args.mr_pipeline_name,
                PipelineParameters=[
                    {
                        'Name' : 'now',
                        'Value' : self.now
                    },
                    {
                        'Name' : 'today',
                        'Value' : self.today
                    },
                    {
                        'Name' : 'model_approval_status',
                        'Value' : 'PendingManualApproval'
                    }
                ]
            )
            print(response)
            
        # pr training pipeline execute if needed 
        if pr_r2 < self.args.pr_r2_threshold or pr_mae > self.args.pr_mae_threshold:
            response = self.client.start_pipeline_execution(
                PipelineName=self.args.pr_pipeline_name,
                PipelineParameters=[
                    {
                        'Name' : 'now',
                        'Value' : self.now
                    },
                    {
                        'Name' : 'today',
                        'Value' : self.today
                    },
                    {
                        'Name' : 'model_approval_status',
                        'Value' : 'PendingManualApproval'
                    }
                ]
            )
            print(response)

    def execution(self):
        # label과 prediction 불러오기
        self._read_label()
        self._read_predictions()
        # monitoring 결과 반환 및 기준 미달 시 재학습 실행
        self.logic()
        print('monitoring process is completed')

    def _read_predictions(self):
        '''
        날짜별 prediction data를 하나의 dataframe으로 합침
        '''
        # 전체 날짜 가지고 오기
        list_dir = os.listdir(os.path.join(self.args.data_path,'input','predictions'))
        min_date = self.min_date.replace('-','')
        max_date = self.max_date.replace('-','')
        df = pd.DataFrame(columns=['date','lounge_type','time_group','pred'])

        inference_count = 0
        inference_date_list = []
        # 조건에 해당 시 통합 dataframe에 concat 진행
        for dir in list_dir:
            if dir>=min_date and dir<max_date:
                prediction = pd.read_csv(os.path.join(self.args.data_path,'input','predictions',dir,self.args.data_name))
                prediction = prediction[prediction['date']==prediction['date'].min()]
                df = pd.concat([df, prediction])
                inference_count+=1
                inference_date_list.append(dir)
        self.df = df
        print('Reading prediction data is completed')
        print(f'inference date count : {inference_count}')
        print(f'date list : {inference_date_list}')

        # 지난 1주일간 inference date가 3일이 넘지 않는다면 알림 제공
        if inference_count<=3:
            self.sns_client.publish(
                TargetArn=self.args.sns_arn,
                Message=f'지난 1주일간 inference date 수 : {inference_count}',
                Subject=f'{self.args.project_name} : input data 적재 이슈 확인'
            )

    def _read_label(self):
        '''
        label 데이터를 불러오고 추가되지 않은 label 데이터 추가 및 전처리 진행
        '''
        # read csv file
        self.label = pd.read_csv(os.path.join(self.args.data_path, 'input', 'label', self.args.label_name))
        
        # label 데이터에서 max date 가지고 오기
        start_date = datetime.strftime(datetime.strptime(self.label['_date'].max(),'%Y-%m-%d'),'%Y%m%d')
        # daily로 쌓인 전체 날짜 폴더명 가지고 오기
        total_dates = os.listdir(os.path.join(self.args.data_path, 'input', 'labels'))
        # 날짜 폴더명에서 start_date보다 큰 날짜만 using date로 반환
        using_dates = [date for date in total_dates if date>=start_date]

        for date in using_dates:
            temp = pd.read_csv(os.path.join(self.args.data_path, 'input', 'labels', date, self.args.daily_label_name))
            self.label = self.label[~self.label['_date'].isin(temp['_date'].unique())]
            # concat
            self.label = pd.concat([self.label, temp])

        # cut date : 30개월 이전 데이터 제거
        self.label['std'] = pd.to_datetime(self.label['_date']).dt.date
        cut_date = self.label['std'].max() - relativedelta(months=30)

        self.label = self.label[self.label['std']>=cut_date]
        self.label = self.label.drop(columns = ['std'])

        # save the data : 추가된 label 데이터 저장
        self.label.to_csv(os.path.join(self.args.data_path, 'output', 'label', self.args.label_name), index=False)
        # label에 있는 최신 날짜(max_date)와 최신 날짜 일주일 전(min_date) 변수 생성
        self.max_date = self.label['_date'].max()
        self.min_date = datetime.strftime(datetime.strptime(self.max_date,'%Y-%m-%d')-relativedelta(days=7),'%Y-%m-%d')
        # monitoring 시 필요한 데이터만을 반환
        self.label = self.label[(self.label['_date']>self.min_date)&(self.label['_date']<=self.max_date)]
        self.label = self.label.reset_index(drop=True)
        # label 데이터 preprocessing
        self.label = self._preprocess_label(self.label)
        print('reading label data is completed')
        print(f'Date added : {using_dates}')



    def _preprocess_label(self, df):
        '''
        label 데이터 preprocessing
        '''
        df['date'] = df['_date']
        
        df['FR'] = df.loc[(df['lng_type'] == 'FR'), 'ke'].astype('int16')
        df['MR'] = df.loc[(df['lng_type'] == 'MR'), 'ke'].astype('int16')
        df['PR'] = df.loc[(df['lng_type'] == 'PR'), 'ke'].astype('int16')

        df_agg = df.groupby(['time_group', 'date']).agg({
            c : 'sum' for c in ['FR', 'MR', 'PR']
        }).astype('int16').reset_index().set_index('date')
        df_agg = df_agg.reset_index()

        # unpivoting lounge column
        df_agg = pd.melt(df_agg, id_vars=['date','time_group'], var_name='lounge_type', value_name='label')
        return df_agg
    

if __name__=='__main__':
    try:
        parser = argparse.ArgumentParser(description='monitoring')

        parser.add_argument('--data_path', default='/opt/ml/processing')
        parser.add_argument('--data_name', default='pnr.csv')
        parser.add_argument('--label_name', default='lounge.csv')
        parser.add_argument('--daily_label_name', default='lounge.000')
        parser.add_argument('--output_name', default='monitoring.csv')
        parser.add_argument('--region_name', default='ap-northeast-2')
        # fr
        parser.add_argument('--fr_pipeline_name', default='sag-dev-ml-lng-fr-train-pipe')
        parser.add_argument('--fr_mae_threshold', default=5)
        # mr
        parser.add_argument('--mr_pipeline_name', default='sag-dev-ml-lng-mr-train-pipe')
        parser.add_argument('--mr_r2_threshold', default=0.5)
        parser.add_argument('--mr_mae_threshold', default=25)
        # pr
        parser.add_argument('--pr_pipeline_name', default='sag-dev-ml-lng-pr-train-pipe')
        parser.add_argument('--pr_r2_threshold', default=0.85)
        parser.add_argument('--pr_mae_threshold', default=100)

        parser.add_argument('--region', type=str, default="ap-northeast-2")
        parser.add_argument('--sns_arn', type=str, default='arn:aws:sns:ap-northeast-2:441848216151:awsdc-sns-dlk-dev-topic-ml-2023p01-lounge')
        parser.add_argument('--project_name', type=str)
        parser.add_argument('--pipeline_name', type=str)
        parser.add_argument('--now', type=str)

        parser.add_argument('--env', type=str)


        args = parser.parse_args()
        monitoring = Monitoring(args)
        monitoring.execution()
    
    except Exception as e:

        publish_sns(
            region_name=args.region,
            sns_arn=args.sns_arn,
            project_name=args.project_name,
            pipeline_name=args.pipeline_name,
            error_type="Monitoring Step 실행 중 에러",
            error_message=e,
            env=args.env,
        )

        raise Exception('step error')