import os
import ast
import time
import boto3
import datetime
import argparse
from pytz import timezone
from pprint import pprint

import sagemaker
from sagemaker.model import ModelPackage


class Deploy():
    
    def __init__(self, args):
        
        self.args = args
        
        self.region = self.args.region
        self.tags = ast.literal_eval(self.args.tags)
        self.role = self.args.role
        self.lounge_name = self.args.lounge_name
        
        self.model_package_group_name = self.args.model_package_group_name
        self.realtime_instance_type = self.args.realtime_instance_type
        self.realtime_initial_instance_count = int(self.args.realtime_initial_instance_count)
        
        self.endpoint_prefix = f"lng-endpoint-{self.lounge_name}"
        self.endpoint_name = None
        
        self.client = boto3.client("sagemaker", region_name=self.region)
        self.sess = sagemaker.session.Session(sagemaker_client=self.client)
        
        
        
    def _get_approved_latest_model_package_arn(self):
        '''
        get latest model package arn within approved status models
        '''
        list_model_packages = self.client.list_model_packages(
            ModelPackageGroupName=self.model_package_group_name
        )['ModelPackageSummaryList']
        list_model_packages.sort(
            key=lambda model_package:model_package['ModelPackageVersion'], 
            reverse=True
        )
        for package in list_model_packages:
            try:
                package['ModelApprovalStatus']=='Approved'
                return package['ModelPackageArn']
            except:
                continue
                
                
    def _get_old_endpoint_name(self):
        '''
        새로운 모델 이전의 endpoint name을 가져옴, 없으면 None Return?
        '''
        endpoints_response = self.client.list_endpoints(
            SortBy="CreationTime",
            SortOrder="Descending",
            NameContains=self.endpoint_prefix,
            StatusEquals="InService",
        )
        
        try:
            old_endpoint_name = endpoints_response["Endpoints"][1]["EndpointName"]
            print("Last Endpoint Name :", old_endpoint_name)
        except:
            old_endpoint_name = False
            print("Couldn't get last endpoint name.")
            print("---Response:")
            pprint(endpoints_response)
        
        return old_endpoint_name
        
        
    
    def _create_endpoint(self):
        
        today = datetime.datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d")
        now = datetime.datetime.now(timezone("Asia/Seoul")).strftime("%H-%M-%S")
        
        self.endpoint_name = f"{self.endpoint_prefix}-{today}-{now}"
        latest_package_arn = self._get_approved_latest_model_package_arn()
        
        self.model = ModelPackage(
            model_package_arn=latest_package_arn,
            role=self.role,
            sagemaker_session=self.sess
        )
        
        self.model.deploy(
            endpoint_name=self.endpoint_name,
            initial_instance_count=self.realtime_initial_instance_count,
            instance_type=self.realtime_instance_type,
            wait=False
        )
        
        # log
        describe_response = self.client.describe_endpoint(
            EndpointName=self.model.endpoint_name
        )
        pprint(describe_response)
        #
        
        while True:
            describe_response = self.client.describe_endpoint(EndpointName=self.model.endpoint_name)
            job_run_status = describe_response['EndpointStatus']
            if job_run_status in ("Failed", "InService", "OutOfService"):
                print("** Job is", job_run_status)
                break

            print(
                describe_response["EndpointStatus"]
            )

            time.sleep(60)
            
            
    
    def _delete_endpoint(self, endpoint_version="old"):
        
        if endpoint_version=="new":
            endpoint_name = self.endpoint_name
            
        else:
            endpoint_name = self._get_old_endpoint_name()
            if not endpoint_name:
                print("No endpoint deleted")
                return
        # log    
        pprint(self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name))
        #
        
    
    
    def execution(self):
        
        self._create_endpoint()
        # self._delete_endpoint("old")



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--region', type=str, default="ap-northeast-2")
    parser.add_argument('--tags', type=str)
    parser.add_argument('--role', type=str)
    parser.add_argument('--lounge_name', type=str)

    parser.add_argument('--model_package_group_name', type=str)
    parser.add_argument('--realtime_instance_type', type=str)
    parser.add_argument('--realtime_initial_instance_count', type=str)
    
    args = parser.parse_args()
    
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    deploy = Deploy(args)
    deploy.execution()