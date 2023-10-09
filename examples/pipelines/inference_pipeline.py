import os
import argparse
from pprint import pprint
from config.config import config_handler
from pytz import timezone
from datetime import datetime

import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TransformStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.transformer import Transformer
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterString

from utils.secret_manager import get_secret
from utils.notification import publish_sns

class InferencePipeline:

    def __init__(self, args):
        # config 변수 가져오기
        self.args = args
        # git id, password 가져오기
        self.secret = get_secret(args.env)
        # sagemaker client 설정
        self.sagemaker_client = boto3.client('sagemaker')
        self._env_setting()

    def _env_setting(self):
        # Lambda를 통해 가져오는 변수 설정
        self.args.today = ParameterString(
            name="today",
            default_value=datetime.strftime(datetime.now(timezone('Asia/Seoul')), '%Y%m%d')
        )
        self.fr_model_name = ParameterString(
            name='fr_model_name',
            default_value=self._get_approved_latest_model_name(self.args.config.get_value('FR-INFERENCING','model_package_group_name'), self.sagemaker_client)
        )
        self.mr_model_name = ParameterString(
            name='mr_model_name',
            default_value=self._get_approved_latest_model_name(self.args.config.get_value('MR-INFERENCING','model_package_group_name'), self.sagemaker_client)
        )
        self.pr_model_name = ParameterString(
            name='pr_model_name',
            default_value=self._get_approved_latest_model_name(self.args.config.get_value('PR-INFERENCING','model_package_group_name'), self.sagemaker_client)
        )
        # 고정 변수 설정
        self.data_bucket = self.args.config.get_value('COMMON','data_bucket')
        self.code_bucket = self.args.config.get_value('COMMON','code_bucket')
        self.prefix = self.args.config.get_value('COMMON','prefix')
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE
        self.base_path = os.path.join(
            's3://',
            self.data_bucket,
            self.prefix
        )

        self.execution_role = self.args.config.get_value("COMMON","role")
        self.pipeline_name = self.args.config.get_value("COMMON","pipeline_name")
        self.tags = self.args.config.get_value('COMMON', 'tags', dtype='list') 
        
        self.cache_config = CacheConfig(
            enable_caching=self.args.config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.args.config.get_value("PIPELINE", "expire_after")
        )

        self.git_config = {
            'repo' : self.args.config.get_value('GIT','git_repo'),
            'branch' : self.args.config.get_value('GIT','git_branch'),
            'username' : self.secret['USER'].split('@')[0],
            'password' : self.secret['PASSWORD'],
        }

        self.pipeline_session = PipelineSession()

    
    def _get_approved_latest_model_package_arn(self, model_package_group_name, sagemaker_client):
        '''
        get latest model package arn within approved status models
        '''
        list_model_packages = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name
        )['ModelPackageSummaryList']
        list_model_packages.sort(
            key = lambda model_package : model_package['ModelPackageVersion'], 
            reverse=True
        )
        for package in list_model_packages:
            if package['ModelApprovalStatus']=='Approved':
                return package['ModelPackageArn']
            continue

    def _get_approved_latest_model_name(self, model_package_group_name, sagemaker_client):
        '''

        '''
        model_arn = self._get_approved_latest_model_package_arn(model_package_group_name, sagemaker_client)
        model_name = sagemaker_client.describe_model_package(
            ModelPackageName=model_arn
        )["ModelPackageDescription"]
        
        return model_name
    
    def _step_preprocess(self):

        # 경로 정의
        prefix_prep = '/opt/ml/processing'
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/warehouse/pnr/(date)
        data_path = Join(
            on = '/',
            values = [
            self.base_path,
            self.args.config.get_value('PREPROCESSING', 'data_path'),
            self.args.today
            ]
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/inference
        target_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PREPROCESSING', 'target_path')
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/etc
        etc_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PREPROCESSING','etc_path')
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/train/raw
        integrated_data_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PREPROCESSING','integrate_data_path')
        )

        # FrameworkProcessor 정의
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value('PREPROCESSING','framework_version'),
            role=self.execution_role,
            instance_type=self.args.config.get_value('PREPROCESSING','instance_type'),
            instance_count=self.args.config.get_value('PREPROCESSING','instance_count', dtype='int'),
            sagemaker_session=self.pipeline_session,
            tags=self.tags
        )
        # step argument 정의
        # input : today data, holiday data, integration data
        # output : fr, mr, pr inference preprocess data, holiday data, integration data
        step_args = prep_processor.run(
            code='./inference_preprocess.py',
            source_dir='./source/preprocess/',
            git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name='today-input',
                    source=data_path,
                    destination=Join(
                        on='/',
                        values=[prefix_prep, self.args.today, 'input'])
                ),
                ProcessingInput(
                    input_name='holiday-input',
                    source=etc_path,
                    destination=os.path.join(prefix_prep, 'etc')
                ),
                ProcessingInput(
                    input_name='integrate-input',
                    source=integrated_data_path,
                    destination=os.path.join(prefix_prep, 'integrate')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='mr-inference-data',
                    source=os.path.join(prefix_prep, 'output', 'mr'),
                    destination=Join(
                        on='/',
                        values=[target_path, self.args.today, 'mr'])
                ),
                ProcessingOutput(
                    output_name='fr-inference-data',
                    source=os.path.join(prefix_prep, 'output', 'fr'),
                    destination=Join(
                        on='/',
                        values=[target_path, self.args.today, 'fr'])
                ),
                ProcessingOutput(
                    output_name='pr-inference-data',
                    source=os.path.join(prefix_prep, 'output', 'pr'),
                    destination=Join(
                        on='/',
                        values=[target_path, self.args.today, 'pr'])
                ),
                ProcessingOutput(
                    output_name='integrated-data',
                    source=os.path.join(prefix_prep, 'output', 'integrate'),
                    destination=integrated_data_path
                ),
                ProcessingOutput(
                    output_name='holiday-output',
                    source=os.path.join(prefix_prep, 'output', 'etc'),
                    destination=etc_path
                )
            ],
            arguments=[
                '--today', self.args.today,
                '--sns_arn', self.args.config.get_value("SNS", "arn"),
                '--project_name', self.args.config.get_value("COMMON", "project_name"),
                '--pipeline_name', self.args.config.get_value("COMMON", "pipeline_name"),
                '--env', self.args.config.get_value("COMMON", "env")
            ],
            job_name='inference_preprocessing'
        )
        # step 정의
        self.preprocessing_process = ProcessingStep(
            name='InferencePreprocessingProcess',
            step_args=step_args,
            cache_config=self.cache_config
        )

        ## logging ##########
        print("  \n== Preprocessing Step ==")
        print("   \nArgs: ")
        for key, value in self.preprocessing_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.preprocessing_process.properties))

    def _step_fr_inference(self):
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/output/(date)/fr
        target_path = Join(
            on='/',
            values = [
            self.base_path,
            self.args.config.get_value('FR-INFERENCING','target_path'),
            self.args.today,
            'fr']
        )
        # model name 기반 Transformer class 정의
        transformer = Transformer(
            model_name=self.fr_model_name,
            instance_type=self.args.config.get_value('FR-INFERENCING', 'instance_type'),
            instance_count=self.args.config.get_value('FR-INFERENCING', 'instance_count', dtype='int'),
            output_path=target_path,
            assemble_with='Line',
            accept='text/csv',
            sagemaker_session=self.pipeline_session,
            tags=self.tags
        )
        # batch job의 input 데이터를 정의
        step_args = transformer.transform(
            data=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["fr-inference-data"].S3Output.S3Uri,
            content_type='text/csv',
            split_type='Line',
            join_source="Input"
        )
        # fr batch job step 생성
        self.fr_inference_process = TransformStep(
            name='FrInferenceProcess',
            step_args=step_args
        )

        ## logging ##########
        print("  \n== FR inference Step ==")
        print("   \nArgs: ")
        for key, value in self.fr_inference_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.fr_inference_process.properties))
        
    def _step_mr_inference(self):
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/output/(date)/mr
        target_path = Join(
            on='/',
            values=[
            self.base_path,
            self.args.config.get_value('MR-INFERENCING','target_path'),
            self.args.today,
            'mr']
        )

        transformer = Transformer(
            model_name=self.mr_model_name,
            instance_type=self.args.config.get_value('MR-INFERENCING', 'instance_type'),
            instance_count=self.args.config.get_value('MR-INFERENCING', 'instance_count', dtype='int'),
            output_path=target_path,
            assemble_with='Line',
            accept='text/csv',
            sagemaker_session=self.pipeline_session,
            tags=self.tags
        )

        step_args = transformer.transform(
            data=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["mr-inference-data"].S3Output.S3Uri,
            content_type='text/csv',
            split_type='Line',
            join_source="Input"
        )

        self.mr_inference_process = TransformStep(
            name='MrInferenceProcess',
            step_args=step_args
        )

        ## logging ##########
        print("  \n== MR inference Step ==")
        print("   \nArgs: ")
        for key, value in self.mr_inference_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.mr_inference_process.properties))

    def _step_pr_inference(self):
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/output/(date)/pr
        target_path = Join(
            on='/',
            values=[
            self.base_path,
            self.args.config.get_value('PR-INFERENCING','target_path'),
            self.args.today,
            'pr']
        )

        transformer = Transformer(
            model_name=self.pr_model_name,
            instance_type=self.args.config.get_value('PR-INFERENCING', 'instance_type'),
            instance_count=self.args.config.get_value('PR-INFERENCING', 'instance_count', dtype='int'),
            output_path=target_path,
            assemble_with='Line',
            accept='text/csv',
            sagemaker_session=self.pipeline_session,
            tags=self.tags
        )

        step_args = transformer.transform(
            data=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["pr-inference-data"].S3Output.S3Uri,
            content_type='text/csv',
            split_type='Line',
            join_source="Input"
        )

        self.pr_inference_process = TransformStep(
            name='PrInferenceProcess',
            step_args=step_args
        )

        ## logging ##########
        print("  \n== PR inference Step ==")
        print("   \nArgs: ")
        for key, value in self.pr_inference_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.pr_inference_process.properties))

    def _step_postprocess(self):
        
        prefix_prep = '/opt/ml/processing'
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/transformed_output/(date)
        target_path = Join(
            on='/',
            values=[
            self.base_path,
            self.args.config.get_value('POSTPROCESSING','target_path'),
            self.args.today]
        )
        # postprocess의 FrameworkProcessor 정의
        post_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value('POSTPROCESSING','framework_version'),
            role=self.execution_role,
            instance_type=self.args.config.get_value('POSTPROCESSING','instance_type'),
            instance_count=self.args.config.get_value('POSTPROCESSING','instance_count', dtype='int'),
            sagemaker_session=self.pipeline_session,
            tags=self.tags,
            output_kms_key = self.args.config.get_value('POSTPROCESSING','kms_key')
        )
        # step argument 정의
        # input : fr, pr, mr inference data
        # output : fr, pr, mr 통합 데이터
        step_args = post_processor.run(
            code='./postprocess.py',
            source_dir='./source/postprocess/',
            git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name='fr-input',
                    source=self.fr_inference_process.properties.TransformOutput.S3OutputPath,
                    destination=os.path.join(prefix_prep, 'input', 'fr')
                ),
                ProcessingInput(
                    input_name='mr-input',
                    source=self.mr_inference_process.properties.TransformOutput.S3OutputPath,
                    destination=os.path.join(prefix_prep, 'input', 'mr')
                ),
                ProcessingInput(
                    input_name='pr-input',
                    source=self.pr_inference_process.properties.TransformOutput.S3OutputPath,
                    destination=os.path.join(prefix_prep, 'input', 'pr')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="output",
                    source=os.path.join(prefix_prep, "output"),
                    destination=target_path,
                ),
            ],
            arguments=[
                '--region', self.args.config.get_value("COMMON", "region"),
                '--sns_arn', self.args.config.get_value("SNS", "arn"),
                '--project_name', self.args.config.get_value("COMMON", "project_name"),
                '--pipeline_name', self.args.config.get_value("COMMON", "pipeline_name"),
                '--env', self.args.config.get_value("COMMON", "env")
            ]
        )
        # step 정의
        self.postprocessing_process = ProcessingStep(
            name='InferencePostprocessingProcess',
            step_args=step_args,
            cache_config=self.cache_config
        )

        ## logging ##########
        print("  \n== Postprocessing Step ==")
        print("   \nArgs: ")
        for key, value in self.postprocessing_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.postprocessing_process.properties))

    def _get_pipeline(self):
        # pipeline 클래스 정의
        # pipeline명, 활용 step, session, 변수 정의
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[self.preprocessing_process, self.fr_inference_process, self.mr_inference_process, self.pr_inference_process, self.postprocessing_process],
            sagemaker_session=self.pipeline_session,
            parameters=[self.args.today, self.fr_model_name, self.mr_model_name, self.pr_model_name]
        )
        return pipeline

    def execution(self):
        # step을 실행하여 각각의 step 정의
        self._step_preprocess()
        self._step_fr_inference()
        self._step_mr_inference()
        self._step_pr_inference()
        self._step_postprocess()
        # pipeline 클래스 정의
        pipeline = self._get_pipeline()
        # pipeline upsert
        pipeline.upsert(role_arn=self.execution_role, tags=self.tags)
        # pipeline 실행
        execution = pipeline.start()

        print(execution.describe)

        execution.wait(max_attempts=120, delay=60)

        print("\n#####Execution completed. Execution step details:")
        print(execution.list_steps())
        
if __name__=='__main__':
    try:
        # 파일 실행 위치 변경
        strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
        os.chdir(strBasePath)
        # get config and argument
        parser = argparse.ArgumentParser()
        # 환경 정의(dev, prd)
        parser.add_argument('--env', default='dev')
        args, _ = parser.parse_known_args()
        # 불러올 config 파일 설정
        if args.env=='dev': 
            config_file = "dev_inference_config.ini"
        elif args.env=='prd':
            config_file = "prd_inference_config.ini"
        args.config = config_handler(config_file)

        # execute monitoring pipeline
        pipe_monitor = InferencePipeline(args)
        pipe_monitor.execution()

        

    except Exception as e:
        # 에러 발생 시, sns 알림 제공
        publish_sns(region_name=args.config.get_value('COMMON','region'),
                    sns_arn=args.config.get_value('SNS','arn'),
                    project_name=args.config.get_value('COMMON','project_name'),
                    pipeline_name=args.config.get_value('COMMON','pipeline_name'),
                    error_type="Inference pipeline 실행 중 에러",
                    error_message=e,
                    env=args.config.get_value('COMMON','env')
                    )
        
        raise Exception('pipeline error')