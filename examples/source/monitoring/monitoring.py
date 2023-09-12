import os
import argparse
import json
import boto3
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from sklearn.metrics import r2_score

class Monitoring:
    
    def __init__(self, args):
        self.args = args
        
        self.client = boto3.client('sagemaker', region_name=self.args.region_name)

    def logic(self):

        self.df = self.df.merge(self.label, on=['date','time_group','lounge_type'], how='inner')

        # fr
        fr = self.df[self.df['lounge_type']=='FR']
        fr_r2 = r2_score(fr['pred'], fr['label'])
        # mr
        mr = self.df[self.df['lounge_type']=='MR']
        mr_r2 = r2_score(mr['pred'], mr['label'])
        # pr
        pr = self.df[self.df['lounge_type']=='PR']
        pr_r2 = r2_score(pr['pred'], pr['label'])

        report_dict = {
            "r2" :
                {
                    'FR' : fr_r2,
                    'MR' : mr_r2,
                    'PR' : pr_r2
                }
        }
        print(f'r2 result : {report_dict}')

        # fr training pipeline execute if needed
        if fr_r2 < self.args.fr_threshold:
            response = self.client.start_pipeline_execution(
                PipelineName=self.args.fr_pipeline_name
            )
            print(response)
        
        # mr training pipeline execute if needed
        if mr_r2 < self.args.mr_threshold:
            response = self.client.start_pipeline_execution(
                PipelineName=self.args.mr_pipeline_name
            )
            print(response)
        
        # pr training pipeline execute if needed 
        if pr_r2 < self.args.pr_threshold:
            response = self.client.start_pipeline_execution(
                PipelineName=self.args.pr_pipeline_name
            )
            print(response)

        # save monitoring data
        output_path = os.path.join(self.args.data_path, 'output', 'monitoring', self.args.output_name)
        with open(output_path, 'w') as f:
            f.write(json.dumps(report_dict))

    def execution(self):
        self._read_label()
        self._read_predictions()

        self.logic()

    def _read_predictions(self):
        list_dir = os.listdir(os.path.join(self.args.data_path,'input','predictions'))
        min_date = self.min_date.replace('-','')
        max_date = self.max_date.replace('-','')
        df = pd.DataFrame(columns=['date','lounge_type','time_group','pred'])

        for dir in list_dir:
            if dir>=min_date and dir<max_date:
                prediction = pd.read_csv(os.path.join(self.args.data_path,'input','predictions',dir,self.args.data_name))
                prediction = prediction[prediction['date']==prediction['date'].min()]
                df = pd.concat([df, prediction])
        self.df = df

    def _read_label(self):
        ## 여기서 이전 7일간의 데이터를 가져오는 방식으로 변경
        # read csv file
        self.label = pd.read_csv(os.path.join(self.args.data_path, 'input', 'label', self.args.label_name))
        
        # get date
        start_date = datetime.strftime(datetime.strptime(self.label['_date'].max(),'%Y-%m-%d'),'%Y%m%d')
        
        total_dates = os.listdir(os.path.join(self.args.data_path, 'input', 'labels'))
        using_dates = [date for date in total_dates if date>=start_date]

        for date in using_dates:
            temp = pd.read_csv(os.path.join(self.args.data_path, 'input', 'labels', date, self.args.daily_label_name))
            self.label = self.label[~self.label['_date'].isin(temp['_date'].unique())]
            # concat
            self.label = pd.concat([self.label, temp])

        # save the data
        self.label.to_csv(os.path.join(self.args.data_path, 'output', 'label', self.args.label_name))
            
        self.max_date = self.label['_date'].max()
        self.min_date = datetime.strftime(datetime.strptime(self.max_date,'%Y-%m-%d')-relativedelta(days=7),'%Y-%m-%d')

        self.label = self.label[(self.label['_date']>self.min_date)&(self.label['_date']<=self.max_date)]
        self.label = self._preprocess_label(self.label)

    def _preprocess_label(self, df):
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
    parser = argparse.ArgumentParser(description='monitoring')
    
    parser.add_argument('--data_path', default='/opt/ml/processing')
    parser.add_argument('--data_name', default='pnr.csv')
    parser.add_argument('--label_name', default='lounge.csv')
    parser.add_argument('--daily_label_name', default='lounge.000')
    parser.add_argument('--output_name', default='monitoring.json')
    parser.add_argument('--region_name', default='ap-northeast-2')
    parser.add_argument('--today', default='20230725')
    # fr
    parser.add_argument('--fr_pipeline_name', default='sag-dev-ml-lng-fr-train-pipe')
    parser.add_argument('--fr_threshold', default=1)
    # mr
    parser.add_argument('--mr_pipeline_name', default='sag-dev-ml-lng-mr-train-pipe')
    parser.add_argument('--mr_threshold', default=0.95)
    # pr
    parser.add_argument('--pr_pipeline_name', default='sag-dev-ml-lng-pr-train-pipe')
    parser.add_argument('--pr_threshold', default=0.95)

    args = parser.parse_args()
    monitoring = Monitoring(args)
    monitoring.execution()