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

from secret_manager.secret_manager import get_secret

class InferencePipeline:

    def __init__(self, args):
        self.args = args
        
        self.secret = get_secret()
        self._env_setting()
        self.sagemaker_client = boto3.client('sagemaker')

    def _env_setting(self):
        
        self.data_bucket = self.args.config.get_value('COMMON','data_bucket')
        self.code_bucket = self.args.config.get_value('COMMON','code_bucket')
        self.prefix = self.args.config.get_value('COMMON','prefix')
        # s3://
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

    def _get_approved_latest_model_package_arn(self, model_package_group_name):
        '''
        get latest model package arn within approved status models
        '''
        list_model_packages = self.sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name
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

    def _step_preprocess(self):

        pipeline_session = PipelineSession()
        # 경로 정의
        prefix_prep = '/opt/ml/processing'
        data_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PREPROCESSING', 'data_path'),
            self.args.today
        )
        target_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PREPROCESSING', 'target_path')
        )
        etc_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PREPROCESSING','etc_path')
        )
        integrated_data_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PREPROCESSING','integrate_data_path')
        )

        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value('PREPROCESSING','framework_version'),
            role=self.execution_role,
            instance_type=self.args.config.get_value('PREPROCESSING','instance_type'),
            instance_count=self.args.config.get_value('PREPROCESSING','instance_count', dtype='int'),
            sagemaker_session=pipeline_session,
            tags=self.tags
        )

        step_args = prep_processor.run(
            code='./inference_preprocess.py',
            source_dir='./source/preprocess/',
            git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name='today-input',
                    source=data_path,
                    destination=os.path.join(prefix_prep, self.args.today, 'input')
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
                    destination=os.path.join(target_path, self.args.today, 'mr')
                ),
                ProcessingOutput(
                    output_name='fr-inference-data',
                    source=os.path.join(prefix_prep, 'output', 'fr'),
                    destination=os.path.join(target_path, self.args.today, 'fr')
                ),
                ProcessingOutput(
                    output_name='pr-inference-data',
                    source=os.path.join(prefix_prep, 'output', 'pr'),
                    destination=os.path.join(target_path, self.args.today, 'pr')
                ),
                ProcessingOutput(
                    output_name='integrated-data',
                    source=os.path.join(prefix_prep, 'output', 'integrate'),
                    destination=integrated_data_path
                )
            ],
            arguments=['--today', self.args.today],
            job_name='inference_preprocessing'
        )

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

        pipeline_session = PipelineSession()

        fr_model_grp_nm = self.args.config.get_value('FR-INFERENCING','model_package_group_name')
        fr_model_arn = self._get_approved_latest_model_package_arn(fr_model_grp_nm)

        model_name = self.sagemaker_client.describe_model_package(
            ModelPackageName=fr_model_arn
        )["ModelPackageDescription"]
        

        # boto3가 아니라, SageMaker AutoML을 활용해 학습했을 경우의 model_name 불러오는 방법 : 아래의 코드를 주석 해제하고 코드 실행
        '''
        model_name_prefix = self.sagemaker_client.describe_model_package(
            ModelPackageName=fr_model_arn
        )["ModelPackageDescription"]

        model_name = sagemaker_client.list_models(
            SortBy="CreationTime",
            SortOrder="Descending",
            NameContains=model_name_prefix,
            MaxResults=1,
        )['Models'][0]['ModelName']
        '''
        target_path = os.path.join(
            self.base_path,
            self.args.config.get_value('FR-INFERENCING','target_path'),
            self.args.today,
            'fr'
        )

        transformer = Transformer(
            model_name=model_name,
            instance_type=self.args.config.get_value('FR-INFERENCING', 'instance_type'),
            instance_count=self.args.config.get_value('FR-INFERENCING', 'instance_count', dtype='int'),
            output_path=target_path,
            assemble_with='Line',
            accept='text/csv',
            sagemaker_session=pipeline_session,
            tags=self.tags
        )

        step_args = transformer.transform(
            data=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["fr-inference-data"].S3Output.S3Uri,
            content_type='text/csv',
            split_type='Line',
            join_source="Input"
        )

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
        
        pipeline_session = PipelineSession()

        mr_model_grp_nm = self.args.config.get_value('MR-INFERENCING','model_package_group_name')
        mr_model_arn = self._get_approved_latest_model_package_arn(mr_model_grp_nm)

        model_name = self.sagemaker_client.describe_model_package(
            ModelPackageName=mr_model_arn
        )["ModelPackageDescription"]
        
        target_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MR-INFERENCING','target_path'),
            self.args.today,
            'mr'
        )

        transformer = Transformer(
            model_name=model_name,
            instance_type=self.args.config.get_value('MR-INFERENCING', 'instance_type'),
            instance_count=self.args.config.get_value('MR-INFERENCING', 'instance_count', dtype='int'),
            output_path=target_path,
            assemble_with='Line',
            accept='text/csv',
            sagemaker_session=pipeline_session,
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
        
        pipeline_session = PipelineSession()

        pr_model_grp_nm = self.args.config.get_value('PR-INFERENCING','model_package_group_name')
        pr_model_arn = self._get_approved_latest_model_package_arn(pr_model_grp_nm)

        model_name = self.sagemaker_client.describe_model_package(
            ModelPackageName=pr_model_arn
        )["ModelPackageDescription"]
        
        target_path = os.path.join(
            self.base_path,
            self.args.config.get_value('PR-INFERENCING','target_path'),
            self.args.today,
            'pr'
        )

        transformer = Transformer(
            model_name=model_name,
            instance_type=self.args.config.get_value('PR-INFERENCING', 'instance_type'),
            instance_count=self.args.config.get_value('PR-INFERENCING', 'instance_count', dtype='int'),
            output_path=target_path,
            assemble_with='Line',
            accept='text/csv',
            sagemaker_session=pipeline_session,
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
        
        pipeline_session = PipelineSession()

        prefix_prep = '/opt/ml/processing'
        target_path = os.path.join(
            self.base_path,
            self.args.config.get_value('POSTPROCESSING','target_path'),
            self.args.today
        )

        post_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value('POSTPROCESSING','framework_version'),
            role=self.execution_role,
            instance_type=self.args.config.get_value('POSTPROCESSING','instance_type'),
            instance_count=self.args.config.get_value('POSTPROCESSING','instance_count', dtype='int'),
            sagemaker_session=pipeline_session,
            tags=self.tags
        )

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
            ]
        )

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
        
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[self.preprocessing_process, self.fr_inference_process, self.mr_inference_process, self.pr_inference_process, self.postprocessing_process],
            sagemaker_session=self.pipeline_session
        )
        return pipeline

    def execution(self):
        
        self._step_preprocess()
        self._step_fr_inference()
        self._step_mr_inference()
        self._step_pr_inference()
        self._step_postprocess()

        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.execution_role)
        execution = pipeline.start()

        print(execution.describe)

if __name__=='__main__':
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    # get today date
    today = datetime.strftime(datetime.now(timezone('Asia/Seoul')), '%Y%m%d')

    # get config and argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', default=today)
    args, _ = parser.parse_known_args()

    # args.config = config_handler('prd_inference_config.ini')
    args.config = config_handler('dev_inference_config.ini')

    # execute monitoring pipeline
    pipe_monitor = InferencePipeline(args)
    pipe_monitor.execution()