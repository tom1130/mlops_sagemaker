import os
import io
import boto3
import time
import json
import pandas as pd
import argparse
import ast
from pytz import timezone
from datetime import datetime
from botocore.exceptions import ClientError
from sklearn.metrics import r2_score, mean_absolute_error
from pprint import pprint

from utils.notification import *


class AutoMLBoto3():
    '''
    AutoMLBoto3 클래스 설명 
        Preprocess 이후의 데이터를 활용하여 Training, Evaluation 및 Model Register 과정을 실행함
        
        `__init__` :
            {lounge}_training_config.ini 파일에서 arguments를 전달받음

        `execution` :
            logic_fit()
            logic_create()
            logic_transform()
            logic_evaluate()
            logic_register()
            를 순차적으로 실행

        `logic_fit` :
            CreateAutoMLJobV2 API를 실행하고, DescribeAutoMLJobV2를 주기적으로 실행하여
            AutoMLJob의 Status를 확인하며 'Failed', 'Stopped', 'Completed'가 나올 때까지 waiting

        `logic_create` :
            logic_fit 이후 생성된 최고 성능 모델의 정보를 받아와서 CreateModel API를 실행하고, model_name을 지정

        `logic_transform` :
            CreateTransformJob API를 실행.
            test data를 추론하고 결과값을 저장

        `logic_evaluate` :
            scikit-learn metric(r2_score, mean_absolute_error)를 사용하여, test data 추론 결과와 
            실제값의 evaluation 진행

        `logic_register` : 
            RegisterModel API 활용하여 최고 성능 모델을 Model Registry에 등록
    '''
    def __init__(self, raw_args):
        
        self.raw_args = raw_args
        self.env = self.raw_args.env
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
        self.lounge_name = self.raw_args.lounge_name
        self.today = self.raw_args.today
        self.now = self.raw_args.now
        self.model_approval_status = self.raw_args.model_approval_status
        self.region = self.raw_args.region
        self.execution_role = self.raw_args.role
        
        self.sns_arn = self.raw_args.sns_arn
        self.project_name = self.raw_args.project_name
        self.pipeline_name = self.raw_args.pipeline_name

        self.auto_ml_job_name = f"sag-{self.env}-ml-lng-{self.lounge_name}-{self.today}{self.now}"
        self.transform_job_name = f"sag-{self.env}-ml-lng-{self.lounge_name}-transform-{self.today}{self.now}"
        self.model_name = f"sag-{self.env}-ml-lng-{self.lounge_name}-model-{self.today}{self.now}"
        self.inference_containers = None

        self.metrics = dict(
            Metrics=dict()
        )

        self.args = self._transform_args()

        self.client = boto3.client("sagemaker", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        self.sns = boto3.client("sns", region_name=self.region)



    def _transform_args(self):
        '''
        `__init__` 함수에서 입력받은 args를 SageMaker Boto3 Client가 실행하는 API의
        Parameter 형식에 맞게 변형하여 dictionary 형태로 리턴

        함수 내 정의되는 Parameter들 설명
            - data path : 각각의 Job에서 input으로 활용될 데이터의 base path 지정
            - output path : 각각의 Job에서 output으로 나올 데이터(및 모델)의 base path 지정
            - automl job config : CreateAutoMLJobV2 API 필요 Parameters
            - transform job config : CreateTransformJob API 필요 Parameters
            - register job config : CreateModelPackage API 필요 Parameters
            - evaluate job config : Evaluation 시 필요한 Parameters
            - tags : 대한항공 Billing 관리 위한 tag
        '''
        
        # data path
        train_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/train-data/pnr.csv") 
        validation_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/validation-data/pnr.csv")
        test_target_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/test-data/lounge.csv")
        test_features_path = os.path.join(self.data_base_path, f"train/preprocess/{self.lounge_name}/test-data/pnr.csv")

        # output path
        model_output_path = os.path.join(self.code_base_path, f"model/{self.lounge_name}/{self.today}")
        eval_pred_path = os.path.join(self.data_base_path, f"train/evaluation/{self.lounge_name}/{self.today}")
        eval_output_path = os.path.join(self.code_base_path, f"evaluation/{self.lounge_name}/{self.today}")
        
        # automl job config
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
                    'MaxAutoMLJobRuntimeInSeconds': eval(self.raw_args.max_runtime_for_auto_ml_job), # 8 * 60 * 60
                },
                'Mode': self.raw_args.mode,
                'GenerateCandidateDefinitionsOnly': False,
                'ProblemType': 'Regression',
                'TargetAttributeName': self.raw_args.target_attribute
            }
        }
        auto_ml_job_objective = {'MetricName': self.raw_args.job_objective}

        
        # transform job config
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
            'S3OutputPath': eval_pred_path,
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
        evaluate_pred_path = os.path.join(eval_pred_path, "pnr.csv.out")
        
        
        # tags
        tags = ast.literal_eval(self.raw_args.tags)

        
        # return args in dictionary
        args = dict(
            
            auto_ml_input_data_config=auto_ml_input_data_config,
            auto_ml_output_data_config=auto_ml_output_data_config,
            auto_ml_problem_type_config=auto_ml_problem_type_config,
            auto_ml_job_objective=auto_ml_job_objective,
            
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
    


    def _get_best_candidate(self):
        '''
        `logic_fit` 함수 실행 이후, DescribeAutoMLJobV2 API의 Response에서 Best Candidate(최고 성능 모델) 정보를 가져옴

        상세 설명
            - DescribeAutoMLJobV2 API에서 Response를 받아, self.inference_containers 변수에 
              Best Candidate의 Inference Containers 정보를 저장
            - self.inference_containers 변수는, Transform Job을 만들기 위한 CreateModel API에서 활용될 것
        '''
        try:
            best_candidate = self.client.describe_auto_ml_job_v2(
                AutoMLJobName=self.auto_ml_job_name
            )['BestCandidate']

        except KeyError:
            # Response에 "BestCandidate" Key가 없는 경우의 error handling 부분.
            # AutoMLJob이 모델을 생성하기 위한 충분한 시간이 없었을 가능성이 있음.
            # SNS 통해 Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Get Best Candidate 실행 중 에러",
                error_message="'BestCandidate' Key not exist, best candidate may not be generated.",
                env=self.env,
            )
            raise

        except ClientError as error:
            # boto3.client Error 발생 시, SNS 통해 Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Get Best Candidate 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )
            raise
        
        self.inference_containers = best_candidate['InferenceContainers']
        
        
        
    def _create_model_package_group(self):
        '''
        CreateModelPackage API 실행 전, 지정한 Model Package Group Name을 가진 Model Package Group이 있는지 
        확인하고 없을 경우 신규 Model Package Group을 생성함
        '''
        try:
            # 지정한 Model Package Group Name을 가진 Model Package Group을 생성
            self.client.create_model_package_group(
                ModelPackageGroupName=self.args['model_package_group_name'],
                Tags=self.args['tags'],
            )
            # log
            comment = f"Created Model Package Group : {self.args['model_package_group_name']}"
            print(comment)
            
        except ClientError as error:
            # Model Package Group이 이미 존재할 경우, existing Model Package Group을 사용
            print("Use existing model package group")
    


    def _reject_pending_model_packages(self):
        '''
        ModelApprovalStatus == "PendingManualApproval" 상태인 Model Package의 상태를 rejected 상태로 변경

        상세 설명
            - ListModelPackages API의 Response에서, ModelApprovalStatus=="PendingManualApproval"인 Model만 가져옴
            - Model들의 Arn List를 생성하여, for문을 돌며 Model을 하나씩 업데이트(PendingManualApproval -> Rejected)
        '''
        try:
            # Pending model list 생성
            pending_model_package_list = self.client.list_model_packages(
                SortBy='CreationTime',
                SortOrder='Descending',
                ModelApprovalStatus='PendingManualApproval',
                ModelPackageGroupName=self.args['model_package_group_name'],
            )['ModelPackageSummaryList']

            if len(pending_model_package_list)==0:
                # Pending Model이 없다면, 함수를 pass
                print('No pending models.')

            else:
                # Model의 Arn만 추출하여 list 생성
                pending_model_package_arns = [model['ModelPackageArn'] for model in pending_model_package_list]
                
                print("Pending Models :")
                pprint(pending_model_package_arns)

                for arn in pending_model_package_arns:
                    # pending model arn list를 for문으로 돌며 UpdateModelPackage API 실행
                    try:
                        reject_response = self.client.update_model_package(
                            ModelPackageArn=arn,
                            ModelApprovalStatus="Rejected",
                            ApprovalDescription=f"[by pipeline] no approval by ai/ml team. {datetime.now()}"
                        )
                        print(f"Rejected : {arn}")

                    except ClientError as error:
                        # Pending Model 업데이트 중 오류가 날 경우, Error 발생을 알리고 Job은 계속 진행
                        # boto3.client Error 발생 시 SNS 통해 Publish
                        publish_sns(
                            region_name=self.region,
                            sns_arn=self.sns_arn,
                            project_name=self.project_name,
                            pipeline_name=self.pipeline_name,
                            error_type="Delete Pending Model 실행 중 에러",
                            error_message=error.response['Error']['Message'],
                            env=self.env,
                        )
                        print(f"Error deleting pending model : {arn}")

        except ClientError as error:
            # Pending Model의 list를 가져오는 데에서 오류가 날 경우, Error 발생을 알리고 Job은 계속 진행
            # boto3.client Error 발생 시 SNS 통해 Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="List pending models to delete 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )
            print('Error listing pending models')

    
    def _csv_from_s3(self, filepath, header='infer'):
        '''
        csv파일을 s3에서 로드하여 pd.DataFrame 객체를 리턴
        boto3의 s3 client 활용

        Args
            filepath : csv 파일의 s3 path
            header : pd.read_csv 파라미터 중 하나.
                =='infer' : 컬럼 정보를 자동 추론
                =='None' : csv 파일에 컬럼을 위한 row가 없음
                ==<type:int> : csv 파일의 <int> 번째 row에 컬럼 정보가 기록되어 있음
        Return
            pandas.DataFrame 객체
        '''
        bucket_name = filepath[5:].split('/')[0]
        start_index = len("s3://") + len(bucket_name) + len("/")
        key = filepath[start_index:]
        
        # log
        print(f"Load csv file from s3 - Bucket : {bucket_name}, Key : {key}, Header : {header}")
        #
        try:
            # boto3 s3 client 활용
            obj = self.s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()), header=header)

        except ClientError as error:
            # csv 파일 로딩에서 에러 발생 시, Evaluation Job 실행이 불가능하므로 pipeline 정지
            # boto3.client Error 발생 시 SNS 통해 Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Load csv file from s3 bucket 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )
            raise

        return df
        
    
    
    def _json_to_s3(self, filepath, file):
        '''
        dictionary 형태 객체를 json으로 변환
        json file을 evaluation_metrics.json 이라는 이름으로, 지정한 s3 path에 업로드
        boto3의 s3 client 활용

        Args
            filepath : evaluation_metrics.json 파일이 생성될 s3 위치
            file : evaluation_metrics를 담고 있는 dictionary 객체
        Return
            None
        '''
        bucket_name = filepath[5:].split('/')[0]
        start_index = len("s3://") + len(bucket_name) + len("/")
        key = os.path.join(filepath[start_index:], "evaluation_metrics.json")
        
        encode_file = json.dumps(file, indent=4, ensure_ascii=False)
        try:
            # boto3 s3 client 활용
            self.s3.put_object(Bucket=bucket_name, Key=key, Body=encode_file)
            print(f"Save json file to s3 - Bucket : {bucket_name}, Key : {key}")
        
        except ClientError as error:
            # json file 업로드 중 에러 발생 시, 모델 학습 및 등록에는 문제가 없으므로 pipeline은 지속함
            # boto3.client Error 발생 시 SNS 통해 Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Save json file to s3 bucket 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )

    
    
    def logic_fit(self, ):
        '''
        AutoML Job을 실행하여 모델 아티팩트 생성

        상세 설명
            CreateAutoMLJobV2 API 실행
                - _transform_args()의 automl job config를 parameters로 활용
                - self.auto_ml_job_name : 오늘 날짜 및 파이프라인 실행 시간을 담아서 생성된 Unique Job Name. 
                  중복된 이름의 Job이 있으면 오류 발생
            Validation Metrics 저장
                - DescribeAutoMLJobV2 실행하여, validation data set에서의 Metric Scores를 self.metrics에 저장
        '''
        try:
            create_auto_ml_response = self.client.create_auto_ml_job_v2(
                AutoMLJobName=self.auto_ml_job_name,
                AutoMLJobInputDataConfig=self.args['auto_ml_input_data_config'],
                OutputDataConfig=self.args['auto_ml_output_data_config'],
                AutoMLProblemTypeConfig=self.args['auto_ml_problem_type_config'],
                RoleArn=self.execution_role,
                Tags=self.args['tags'],
                AutoMLJobObjective=self.args['auto_ml_job_objective'],
            )

        except ClientError as error:
            # boto3.client Error 발생 시 SNS 통해 Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Create AutoML Job(V2) step 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )
            raise
        
        # logging을 위해 DescribeAutoMLJobV2 API 수행
        describe_response = self.client.describe_auto_ml_job_v2(AutoMLJobName=self.auto_ml_job_name)
        print("AutoML Job Created(boto3).")
        pprint(describe_response)

        while True:
            # Job 수행 완료될 때까지 waiting
            describe_response = self.client.describe_auto_ml_job_v2(AutoMLJobName=self.auto_ml_job_name)
            job_run_status = describe_response['AutoMLJobStatus']
            if job_run_status in ("Completed", "Stopped"):
                print("** Job is", job_run_status)
                break

            if job_run_status == "Failed":
                # Job이 Fail했을 경우, SNS 통해 Publish
                publish_sns(
                    region_name=self.region,
                    sns_arn=self.sns_arn,
                    project_name=self.project_name,
                    pipeline_name=self.pipeline_name,
                    error_type="Create AutoML Job(V2) step 실행 중 에러",
                    error_message=f"Job is {job_run_status}",
                    env=self.env,
                )
                raise Exception

            print(
                describe_response["AutoMLJobStatus"]
                + " - "
                + describe_response["AutoMLJobSecondaryStatus"]
            )

            time.sleep(60)

        # Validation DataSet에서의 Metric Score 저장
        validation_metric_scores = describe_response['BestCandidate']['CandidateProperties']['CandidateMetrics']
        for metric in validation_metric_scores:
            metric_dict = {
                f"VAL_{metric['MetricName']}":{
                    "value":metric["Value"]
                }
            }
            self.metrics["Metrics"].update(metric_dict)
            print(f"{metric['MetricName']} - Validation Set: {metric['Value']}")

    
        
    def logic_create(self, ):
        '''
        SageMaker Model을 생성

        상세 설명
            _get_best_candidate()
                - 이전에 실행한 AutoMLJob에서 최고 성능 모델(Best Candidate)의 inference_containers 정보를 
                  self.inference_containers 변수에 저장
            CreateModel API 실행
                - self.model_name : 오늘 날짜 및 파이프라인 실행 시간을 담아서 생성된 Unique model name. 
                  중복된 이름의 Model이 있으면 오류 발생
                - self.inference_containers : Best Candidate의 모델 및 이미지 정보
        '''
        
        self._get_best_candidate()

        try:
            create_model_response = self.client.create_model(
                ModelName=self.model_name,
                ExecutionRoleArn=self.execution_role,
                Containers=self.inference_containers,
                Tags=self.args['tags'],
            )
        
        except ClientError as error:
            # boto3.client Error 발생 시 SNS 통해 Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Create Model step 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )
            raise
        
        # log
        print("Model Created(boto3).")
        pprint(create_model_response)
        #
        
        
        
    def logic_transform(self, ):
        '''
        SageMaker Model을 활용하여 Transform Job을 생성

        상세 설명
            CreateTransformJob API 실행
                - _transform_args()의 transform job config를 parameter로 활용
                - self.transform_job_name : 오늘 날짜 및 파이프라인 실행 시간을 담아서 생성된 Unique Job Name. 
                  중복된 이름의 Job이 있을 경우 오류 발생
        '''
        try:
            transform_response = self.client.create_transform_job(
                TransformJobName=self.transform_job_name,
                ModelName=self.model_name,
                BatchStrategy='SingleRecord',
                TransformInput=self.args['transform_input_data_config'],
                TransformOutput=self.args['transform_output_data_config'],
                TransformResources={
                    'InstanceType': self.args['transform_instance_type'],
                    'InstanceCount': self.args['transform_instance_count'],
                },
                Tags=self.args['tags'],
            )

        except ClientError as error:
            # boto3.client Error 발생 시 SNS Publish
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Create Transform job step 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )
            raise   
        
        # logging을 위해 DescribeTransformJob API 수행
        describe_response = self.client.describe_transform_job(TransformJobName=self.transform_job_name)
        print("Transform Job Created(boto3).")
        pprint(describe_response)
        #

        while True:
            # Job 수행 완료될 때까지 waiting
            describe_response = self.client.describe_transform_job(TransformJobName=self.transform_job_name)
            job_run_status = describe_response['TransformJobStatus']
            if job_run_status in ("Completed", "Stopped"):
                print("** Job is", job_run_status)
                break

            if job_run_status == "Failed":
                # Job이 Fail했을 경우, SNS 통해 Publish
                publish_sns(
                    region_name=self.region,
                    sns_arn=self.sns_arn,
                    project_name=self.project_name,
                    pipeline_name=self.pipeline_name,
                    error_type="Create Transform job step 실행 중 에러",
                    error_message=f"Job is {job_run_status}",
                    env=self.env,
                )
                raise Exception

            print(describe_response["TransformJobStatus"])

            time.sleep(60)
            
            
            
    def logic_evaluate(self):
        '''
        Test DataSet을 이용해 metric을 산출
        Validation Data 및 Test Data의 metric scores를 json file로 저장
        '''

        # csv reading : SageMaker Batch Transform Job은 디폴트로 dataframe의 헤더가 없어야 함
        test = self._csv_from_s3(self.args['evaluate_test_path'], header=None)
        pred = self._csv_from_s3(self.args['evaluate_pred_path'], header=None)
        
        self.test_score_r2 = r2_score(test, pred)
        self.test_score_mae = mean_absolute_error(test, pred)
        
        # log
        print("R2  - Test Set :", self.test_score_r2)
        print("MAE - Test Set :", self.test_score_mae)
        #
        
        self.metrics["Metrics"].update(
            {
                "TEST_R2":{
                    "value":self.test_score_r2
                }, 
                "TEST_MAE":{
                    "value":self.test_score_mae
                }
            }
        )
        
        print("Evaluation Metrics :")
        pprint(self.metrics)
        
        # metric score를 s3에 json file 형태로 저장
        self._json_to_s3(self.args['evaluate_output_path'], file=self.metrics)
        
    

    def logic_register(self):
        '''
        SageMaker Model을 Registry에 등록

        상세 설명
            _create_model_package_group()
                - Model Package Group Name을 가진 Model Package Group이 없다면 새로 생성
                - 중복된 이름의 Model Package Group이 있다면, 기존의 Group을 활용
            _reject_pending_model_packages()
                - 현재까지의 model registry에서 ModelApprovalStatus=="PendingManualApproval"인
                  모델 status를 "Rejected" 상태로 변경
            CreateModelPackage API 실행
                - _transform_args()의 register job config를 parameter로 활용
                - ModelPackageDescription : inference_pipeline 실행 시, model name이 있어야 Transform Job 생성 가능
                  따라서 Description에 self.model_name을 string형태로 넣어서 inference_pipeline에서 활용가능토록 함
                - ModelMetrics : 모델 metric score가 기록된 json file
        '''
        self._create_model_package_group()
        self._reject_pending_model_packages()
        
        # Model Register
        try:
            register_response = self.client.create_model_package(
                ModelPackageGroupName=self.args['model_package_group_name'],
                ModelPackageDescription=self.model_name,
                InferenceSpecification={
                    'Containers': self.inference_containers,
                    'SupportedContentTypes': self.args['registry_content_types'],
                    'SupportedResponseMIMETypes': self.args['registry_response_types'],
                    'SupportedRealtimeInferenceInstanceTypes': self.args['registry_realtime_instance_types'],
                    'SupportedTransformInstanceTypes': self.args['registry_transform_instance_types'],
                },
                ModelApprovalStatus=self.model_approval_status,
                ModelMetrics={
                    'ModelQuality': {
                        'Statistics': {
                            'ContentType': 'application/json',
                            'S3Uri': os.path.join(self.args['evaluate_output_path'], "evaluation_metrics.json")
                        },
                    }
                }
            )
        
            # Register시 모델 등록 정보 sns로 전송
            subject = "[AWS] {}({}) - {} Model Registered".format(self.project_name, self.env.upper(), self.lounge_name.upper())
            pipeline = self.pipeline_name
            message = '''
            모델이 등록되었습니다.

            1. Project 명 : {}

            2. Pipeline : {}

            3. Model Package Group : {}

            4. Model Name : {}
            '''.format(self.project_name, pipeline, self.args['model_package_group_name'], self.model_name)
            response = self.sns.publish(
                TargetArn=self.sns_arn,
                Message=message,
                Subject=subject,
            )


        except ClientError as error:
            # boto3.client Error 발생 시 SNS Publishs
            publish_sns(
                region_name=self.region,
                sns_arn=self.sns_arn,
                project_name=self.project_name,
                pipeline_name=self.pipeline_name,
                error_type="Register model step 실행 중 에러",
                error_message=error.response['Error']['Message'],
                env=self.env,
            )
            raise
        
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
    parser.add_argument('--env', type=str)
    parser.add_argument('--today', type=str)
    parser.add_argument('--now', type=str)
    parser.add_argument('--role', type=str, default="arn:aws:iam::441848216151:role/DEV-DATA-ML-ADMIN")
    parser.add_argument('--data_bucket', type=str)
    parser.add_argument('--code_bucket', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--lounge_name', type=str)
    parser.add_argument('--region', type=str, default="ap-northeast-2")
    parser.add_argument('--tags', type=str)
    parser.add_argument('--sns_arn', type=str)
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--pipeline_name', type=str)
    
    # Training arguments
    parser.add_argument('--auto_ml_algorithms', type=str)
    parser.add_argument('--max_candidate', type=int, default=100)
    parser.add_argument('--max_runtime_per_training_job', type=str, default="2*60*60")
    parser.add_argument('--max_runtime_for_auto_ml_job', type=str, default="8*60*60")
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
    parser.add_argument('--model_package_group_name', type=str)
    parser.add_argument('--model_approval_status', type=str)
    
    args = parser.parse_args()
    
    automl = AutoMLBoto3(args)
    automl.execution()