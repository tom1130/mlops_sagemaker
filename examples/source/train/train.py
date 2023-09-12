import os
import io
import boto3
import time
import json
import pandas as pd
import argparse
import ast
from sklearn.metrics import r2_score
from pprint import pprint

class AutoMLBoto3Base():
    
    def __init__(self, raw_args):
        '''
        AutoMLBoto3Base : 사용자 지정 raw arguments를 받아 boto3.client 문법에 맞는 input형태로 변환하는 클래스
        - args : 사용자가 변경하여 사용 가능한 변수
        - vars : 이 코드에서 자동으로 생성되는 변수
        '''
        self.raw_args = raw_args
        self.vars = dict(
            auto_ml_job_name=None,
            model_name=None,
            transform_job_name=None,
        )

        self.lounge_name = self.raw_args.lounge_name
        self.today = self.raw_args.today
        self.now = self.raw_args.now

        self.data_base_path = os.path.join(
            "s3://",
            self.raw_args.data_bucket,
            self.raw_args.prefix
        )

        self.code_base_path = os.path.join(
            "s3://",
            self.raw_args.code_bucket,
            self.raw_args.prefix
        )
    
    
    def _get_variables(self):
        '''
        오늘 날짜 및 현재 시각에 따라 job name을 자동 생성하고, 인스턴스 변수에 저장
        '''
        self.vars['auto_ml_job_name'] = self.lounge_name + "-auto-ml-" + self.today + self.now
        self.vars['transform_job_name'] = self.lounge_name + "-transform-" + self.today + self.now
    
    
    
    def _transform_args(self):
        '''
        입력받은 raw_args를 boto3.client의 문법에 맞는 input형태로 변형하여 dictionary형태의 args로 리턴
        리턴된 args는 아래 AutoMLBoto3Run 클래스에서 logic 함수의 input으로 활용
        '''
        # data path
        train_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/train-data/pnr.csv") 
        validation_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/validation-data/pnr.csv")
        test_target_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/test-data/pnr_target.csv")
        test_features_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/test-data/pnr_drop_target.csv")
        model_output_path = os.path.join(self.code_base_path, f"model/{self.lounge_name}/{self.today}")
        eval_output_path = os.path.join(self.data_base_path, f"train/evaluation/{self.lounge_name}/{self.today}")
        
        # automl job config
        auto_ml_job_name = self.vars['auto_ml_job_name']
        auto_ml_input_data_config = [
            {
                'ChannelType': 'training',
                'ContentType': 'text/csv;header=present',
                'CompressionType': 'None',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': train_path,
                    }
                }
            },
            {
                'ChannelType': 'validation',
                'ContentType': 'text/csv;header=present',
                'CompressionType': 'None',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': validation_path,
                    }
                }
            },
        ]
        auto_ml_output_data_config = {"S3OutputPath": model_output_path}
        auto_ml_problem_type_config = {
            'TabularJobConfig': {
                'CandidateGenerationConfig': {
                    'AlgorithmsConfig': [
                        {
                            'AutoMLAlgorithms': self.raw_args.auto_ml_algorithms.split(),
                        },
                    ]
                },
                'CompletionCriteria': {
                    'MaxCandidates': self.raw_args.max_candidate, # 100,
                    'MaxRuntimePerTrainingJobInSeconds': eval(self.raw_args.max_runtime_per_training_job), # 2 * 60 * 60,
                    'MaxAutoMLJobRuntimeInSeconds': eval(self.raw_args.max_runtime_for_auto_ml_job), # 25 * 60 # 8 * 60 * 60
                },
                'Mode': self.raw_args.mode,
                'GenerateCandidateDefinitionsOnly': False,
                'ProblemType': 'Regression',
                'TargetAttributeName': self.raw_args.target_attribute
            }
        }
        auto_ml_job_objective = {'MetricName': self.raw_args.job_objective}

        
        # transform job config
        transform_job_name=self.vars['transform_job_name']
        transform_input_data_config = {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': test_features_path,
                    }
                },
                'ContentType': 'text/csv',
                'SplitType': 'Line'
            }
        transform_output_data_config = {
            'S3OutputPath': eval_output_path,
            'Accept': 'text/csv',
            'AssembleWith': 'Line',
        }
        transform_instance_type = self.raw_args.transform_instance_type # "ml.m5.large"
        transform_instance_count = self.raw_args.transform_instance_count # 1
        
        
        # register job config
        registry_content_types = [self.raw_args.registry_content_type] # ['text/csv']
        registry_response_types = [self.raw_args.registry_response_type] # ['text/csv']
        registry_realtime_instance_types = [self.raw_args.registry_realtime_instance_type] # ['ml.m5.large']
        registry_transform_instance_types = [self.raw_args.registry_transform_instance_type] # ['ml.m5.large']
        model_package_group_name = self.raw_args.model_package_group_name
        
        
        # evaluate job config
        evaluate_output_path = eval_output_path
        evaluate_test_path = test_target_path
        evaluate_pred_path = os.path.join(eval_output_path, "pnr_drop_target.csv.out")
        
        
        # tags
        tags = ast.literal_eval(self.raw_args.tags)

        
        # return values
        args = dict(
            
            auto_ml_job_name=auto_ml_job_name,
            auto_ml_input_data_config=auto_ml_input_data_config,
            auto_ml_output_data_config=auto_ml_output_data_config,
            auto_ml_problem_type_config=auto_ml_problem_type_config,
            auto_ml_job_objective=auto_ml_job_objective,
            
            transform_job_name=transform_job_name,
            transform_input_data_config=transform_input_data_config,
            transform_output_data_config=transform_output_data_config,
            transform_instance_type=transform_instance_type,
            transform_instance_count=transform_instance_count,
            
            registry_content_types=registry_content_types,
            registry_response_types=registry_response_types,
            registry_realtime_instance_types=registry_realtime_instance_types,
            registry_transform_instance_types=registry_transform_instance_types,
            model_package_group_name=model_package_group_name,
            
            evaluate_output_path=evaluate_output_path,
            evaluate_test_path=evaluate_test_path,
            evaluate_pred_path=evaluate_pred_path,
            
            tags=tags
        )
        print("===AutoML Arguments")
        pprint(args)
        return args




class AutoMLBoto3Run(AutoMLBoto3Base):
    
    def __init__(self, raw_args):
        '''
        vars 자동 생성 및 args를 transform하여 인스턴스 변수에 저장
        best_candidate의 정보(image 및 model.tar.gz위치)를 담은 inference_containers 인스턴스 변수를 정의
        '''
        super().__init__(raw_args)
        self._get_variables()
        self.args = self._transform_args()
        
        self.region = self.raw_args.region
        self.execution_role = self.raw_args.role # self.args.role
        
        self.client = boto3.client("sagemaker", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        
        self.inference_containers = None
        self.metric = None
        
    
    
    def _get_best_candidate(self):
        '''
        boto3 client - describe auto ml job v2 API를 통해, 지정된 auto_ml_job의 best candidate를 가져오고 model_name과 inference_containers를 지정
        '''
        
        best_candidate = self.client.describe_auto_ml_job_v2(
            AutoMLJobName=self.vars['auto_ml_job_name']
        )['BestCandidate']
        
        model_name = self.lounge_name + "-auto-ml-" + self.today + "-" + self.now
        # model_name = best_candidate['CandidateName']
        
        inference_containers = best_candidate['InferenceContainers']
        
        # log
        print("Best Candidate - Inference Containers : ")
        pprint(inference_containers)
        print("Model Name :", model_name)
        #
        
        self.vars['model_name'] = model_name
        self.inference_containers = inference_containers
        
        
        
    def _create_model_package_group(self):
        '''
        boto3 client - create_model_package_group API를 통해, 사용자가 지정한 이름의 model package group이 이미 존재하는지를 확인하고 존재하지 않을 경우 model package group을 생성
        '''
        try:
            self.client.create_model_package_group(
                ModelPackageGroupName=self.args['model_package_group_name'],
                Tags=self.args['tags'],
            )
            
            # log
            print("Created Model Package Group :", self.args['model_package_group_name'])
            #
            
        except:
            
            # log
            print(self.args['model_package_group_name'], ": Model Package Group Exists.")
            #
            
            pass
    

    
    def _csv_from_s3(self, filepath, header='infer'):
        '''
        csv 파일을 s3에서 로드 (logic_evaluate에서 csv파일 다운로드를 위해 활용)
        '''
        bucket_name = filepath[5:].split('/')[0]
        start_index = len("s3://") + len(bucket_name) + len("/")
        key = filepath[start_index:]
        
        # log
        print(f"Load csv from S3 - Bucket : {bucket_name}, Key : {key}, Header : {header}")
        #
        
        obj = self.s3.get_object(Bucket=bucket_name, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()), header=header)
        return df
        
    
    
    def _json_to_s3(self, filepath, file):
        '''
        json file을 s3에 업로드 (evaluation_metrics.json 업로드를 위해 활용)
        '''
        bucket_name = filepath[5:].split('/')[0]
        start_index = len("s3://") + len(bucket_name) + len("/")
        key = os.path.join(filepath[start_index:], "evaluation_metrics.json")
        
        encode_file = json.dumps(file, indent=4, ensure_ascii=False)
        try:
            self.s3.put_object(Bucket=bucket_name, Key=key, Body=encode_file)
            # log
            print(f"Save json to S3 - Bucket : {bucket_name}, Key : {key}")
            #
            return True
        except:
            # log
            print("# Error occured when saving json file to s3")
            print(f"Save json to S3 - Bucket : {bucket_name}, Key : {key}")
            #
            return False

    
    
    def logic_fit(self, ):
        '''
        boto3 client - create_auto_ml_job_v2 API를 통해, AutoML Job을 생성하고 Job이 완료될 때까지 대기
        '''
        
        create_auto_ml_response = self.client.create_auto_ml_job_v2(
            AutoMLJobName=self.vars['auto_ml_job_name'],
            AutoMLJobInputDataConfig=self.args['auto_ml_input_data_config'],
            OutputDataConfig=self.args['auto_ml_output_data_config'],
            AutoMLProblemTypeConfig=self.args['auto_ml_problem_type_config'],
            RoleArn=self.execution_role,
            Tags=self.args['tags'],
            AutoMLJobObjective=self.args['auto_ml_job_objective'],
        )
        
        # log
        describe_response = self.client.describe_auto_ml_job_v2(AutoMLJobName=self.vars['auto_ml_job_name'])
        print("AutoML Job Created(boto3).")
        pprint(describe_response)
        #

        while True:
            describe_response = self.client.describe_auto_ml_job_v2(AutoMLJobName=self.vars['auto_ml_job_name'])
            job_run_status = describe_response['AutoMLJobStatus']
            if job_run_status in ("Failed", "Completed", "Stopped"):
                print("** Job is", job_run_status)
                break

            print(
                describe_response["AutoMLJobStatus"]
                + " - "
                + describe_response["AutoMLJobSecondaryStatus"]
            )

            time.sleep(60)

    
        
    def logic_create(self, ):
        '''
        실행된 AutoML Job의 Best Candidate 정보를 얻어오고,
        boto3 client - create_model API를 통해 SageMaker Model생성
        '''
        
        self._get_best_candidate()

        create_model_response = self.client.create_model(
            ModelName=self.vars['model_name'],
            ExecutionRoleArn=self.execution_role,
            Containers=self.inference_containers,
            Tags=self.args['tags'],
        )
        
        # log
        print("Model Created(boto3).")
        pprint(create_model_response)
        #
        
        
        
    def logic_transform(self, ):
        '''
        boto3 client - create_transform_job API를 통해 Batch Transform Job을 생성하고 Job이 완료될 때까지 대기
        '''

        transform_response = self.client.create_transform_job(
            TransformJobName=self.vars['transform_job_name'],
            ModelName=self.vars['model_name'],
            BatchStrategy='SingleRecord',
            TransformInput=self.args['transform_input_data_config'],
            TransformOutput=self.args['transform_output_data_config'],
            TransformResources={
                'InstanceType': self.args['transform_instance_type'],
                'InstanceCount': self.args['transform_instance_count'],
            },
            Tags=self.args['tags'],
        )
        
        # log
        describe_response = self.client.describe_transform_job(TransformJobName=self.vars['transform_job_name'])
        print("Transform Job Created(boto3).")
        pprint(describe_response)
        #

        while True:
            describe_response = self.client.describe_transform_job(TransformJobName=self.vars['transform_job_name'])
            job_run_status = describe_response['TransformJobStatus']
            if job_run_status in ("Failed", "Completed", "Stopped"):
                print("** Job is", job_run_status)
                break

            print(describe_response["TransformJobStatus"])

            time.sleep(60)
            
            
            
    def logic_evaluate(self):
        '''
        test, pred 데이터로 r2 score 산출하고 인스턴스 변수로 저장
        '''
        test = self._csv_from_s3(self.args['evaluate_test_path'], header=None)
        pred = self._csv_from_s3(self.args['evaluate_pred_path'], header=None)
        
        metric = r2_score(pred, test)
        self.metric = metric
        
        # log
        print("Metric Score :", metric)
        #
        
        evaluation_metrics = {
            "classification_metrics": {
                "r2": {
                    "value": metric, 
                    "standard_deviation": "NaN"
                }
            }
        }
        
        self._json_to_s3(self.args['evaluate_output_path'], file=evaluation_metrics)
        
        
            
    def logic_register(self):
        '''
        model package group을 생성
        boto3 client - create_model_package API를 통해 model package를 생성
        '''

        self._create_model_package_group()

        register_response = self.client.create_model_package(
            ModelPackageGroupName=self.args['model_package_group_name'],
            ModelPackageDescription=self.vars['model_name'],
            InferenceSpecification={
                'Containers': self.inference_containers,
                'SupportedContentTypes': self.args['registry_content_types'],
                'SupportedResponseMIMETypes': self.args['registry_response_types'],
                'SupportedRealtimeInferenceInstanceTypes': self.args['registry_realtime_instance_types'],
                'SupportedTransformInstanceTypes': self.args['registry_transform_instance_types'],
            },
            ModelApprovalStatus="PendingManualApproval",
            ModelMetrics={
                'ModelQuality': {
                    'Statistics': {
                        'ContentType': 'application/json',
                        'S3Uri': os.path.join(self.args['evaluate_output_path'], "evaluation_metrics.json")
                    },
                }
            }
        )
        
        # log
        print("Model is appended in Model Registry.")
        pprint(register_response)
        #

    

    def execution(self):
        
        self.logic_fit()
        self.logic_create()
        self.logic_transform()
        self.logic_evaluate()
        self.logic_register()
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Common arguments
    parser.add_argument('--today', type=str)
    parser.add_argument('--now', type=str)
    parser.add_argument('--role', type=str, default="arn:aws:iam::441848216151:role/DEV-DATA-ML-ADMIN")
    parser.add_argument('--data_bucket', type=str, default="awsdc-s3-dlk-dev-ml")
    parser.add_argument('--code_bucket', type=str, default="awsdc-s3-dlk-dev-ml")
    parser.add_argument('--prefix', type=str, default="tmp/lounge_2")
    parser.add_argument('--lounge_name', type=str, default="fr")
    parser.add_argument('--region', type=str, default="ap-northeast-2")
    parser.add_argument('--tags', type=str)
    
    # Training arguments
    parser.add_argument('--auto_ml_algorithms', type=str)
    parser.add_argument('--max_candidate', type=int, default=100)
    parser.add_argument('--max_runtime_per_training_job', type=str, default="2*60*60")
    parser.add_argument('--max_runtime_for_auto_ml_job', type=str, default="25*60")
    parser.add_argument('--mode', type=str, default="HYPERPARAMETER_TUNING")
    parser.add_argument('--target_attribute', type=str, default="target")
    parser.add_argument('--job_objective', type=str, default="RMSE")
    
    # Transform arguments
    parser.add_argument('--transform_instance_type', type=str, default="ml.m5.large")
    parser.add_argument('--transform_instance_count', type=int, default=1)
    
    # Register arguments
    parser.add_argument('--registry_content_type', type=str, default='text/csv')
    parser.add_argument('--registry_response_type', type=str, default='text/csv')
    parser.add_argument('--registry_realtime_instance_type', type=str, default='ml.m5.large')
    parser.add_argument('--registry_transform_instance_type', type=str, default='ml.m5.large')
    parser.add_argument('--model_package_group_name', type=str, default='fr-model-group-ex0830cns')
    
    args = parser.parse_args()
    
    automl = AutoMLBoto3Run(args)
    automl.execution()