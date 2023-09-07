import os
import boto3
import botocore
import time, datetime
import json
import sagemaker
import pandas as pd
import argparse
from config.config import config_handler
from pprint import pprint
from pytz import timezone

from sagemaker import get_execution_role
from sagemaker import ModelMetrics, MetricsSource
from sagemaker import AutoML
from sagemaker.automl.automl import AutoMLInput
from sagemaker.sklearn import SKLearn
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor


class pipeline_pr():
    '''
    pipeline_fr과 달라진 점.
    1. Preprocessing 단계를 건너뜀(따라서 Train Step에서의 input은 Preprocessing.Properties를 사용 X)
    2. tmp/lounge_2 데이터를 활용할 거라서 path도 그에 따라서 바꿈. 아래는 바뀐 path목록
        - 
        - 
        
    '''
    def __init__(self, args):
        
        self.args = args
        self.strRegionName = self.args.config.get_value("COMMON", "region")
        
        self.exp = True
        self._env_setting()   
        
        
    def _env_setting(self, ):
        
        self.pipeline_session = PipelineSession()
        self.strExecutionRole = self.args.config.get_value("COMMON", "role")
        self.strLoungeName = self.args.config.get_value("COMMON", "lounge_name")
        self.strModelPackageGroupName = self.args.config.get_value("COMMON", "model_package_group_name")
        self.strDataBucketName = self.args.config.get_value("COMMON", "data_bucket")
        self.strCodeBucketName = self.args.config.get_value("COMMON", "code_bucket")
        self.strPipelineName = self.args.config.get_value("COMMON", "pipeline_name") #"-".join([self.strPrefix, self.strModelName])
        
        
        
    def _step_preprocessing(self, ):
        
        strPrefixPrep = '/opt/ml/processing'
        strDataPath = self.args.config.get_value('PREPROCESSING','data_path')
        strTargetPath = self.args.config.get_value('PREPROCESSING','target_path')
        
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("PREPROCESSING", "framework_version"),
            role=self.strExecutionRole,
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count", dtype='int'),
            sagemaker_session=self.pipeline_session,
        )
        
        step_preprocessing_args = prep_processor.run(
            code = './pr_training_preprocess.py',
            source_dir = '../source/preprocess',
            inputs = [
                ProcessingInput(
                    input_name='input',
                    source=strDataPath,
                    destination=os.path.join(strPrefixPrep, 'input')
                ),
                ProcessingInput(
                    input_name='etc',
                    source=self.args.config.get_value('PREPROCESSING','etc_path'),
                    destination=os.path.join(strPrefixPrep, 'etc')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train-data",
                    source=os.path.join(strPrefixPrep, "output", "train"),
                    destination=os.path.join(strTargetPath,'pr','train-data'),
                ),
                ProcessingOutput(
                    output_name="validation-data",
                    source=os.path.join(strPrefixPrep, "output", "validation"),
                    destination=os.path.join(strTargetPath,'pr','validation-data'),
                ),
                ProcessingOutput(
                    output_name="test-data",
                    source=os.path.join(strPrefixPrep, "output", "test"),
                    destination=os.path.join(strTargetPath,'pr','test-data'),
                )
            ]
        )
        
        self.preprocessing_process = ProcessingStep(
            name = "PrTrainPreprocessingProcess",
            step_args = step_preprocessing_args,
            # cache_config = self.cache_config
        )

        ## logging ##########
        print("  \n== Preprocessing Step ==")
        print("   \nArgs: ")
        for key, value in self.preprocessing_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.preprocessing_process.properties))
        
        
    def _get_pipeline(self, ):
                
        pipeline = Pipeline(
            name=self.strPipelineName,
            steps=[
                self.preprocessing_process, 
                # self.training_process, 
                # self.model_creation_process,
                # self.batch_transform_process,
                # self.evaluation_process, 
                # self.model_registration_process,
            ],
            sagemaker_session=self.pipeline_session
        )
    
        return pipeline
    
        
    def execution(self, ):
        
        self._step_preprocessing()
        # self._step_training()
        # self._step_model_creation()
        # self._step_batch_transform()
        # self._step_evaluation()
        # self._step_model_registration()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.strExecutionRole)
        execution = pipeline.start()
        desc = execution.describe()
        
        print(desc)
    
    
if __name__=="__main__":
    
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    args.config = config_handler('pr_train_config.ini')
    
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe = pipeline_pr(args)
    pipe.execution()