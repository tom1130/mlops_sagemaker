import os
import argparse
from pprint import pprint
from config.config import config_handler
from pytz import timezone
from datetime import datetime

import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.pipeline_context import PipelineSession

from secret_manager.secret_manager import get_secret


class ModelEndpointPipeline():
    
    def __init__(self, args):
        self.args = args
        
        self.secret = get_secret()
        self._env_setting()
        self.sagemaker_client = boto3.client('sagemaker')
        
        
    
    def _env_setting(self):
        
        self.pipeline_session = PipelineSession()
        self.execution_role = self.args.config.get_value("COMMON", "role")
        self.pipeline_name = self.args.config.get_value("COMMON","pipeline_name")
        
        self.git_config = {
            'repo' : self.args.config.get_value('GIT','git_repo'),
            'branch' : self.args.config.get_value('GIT','git_branch'),
            'username' : self.secret['USER'].split('@')[0],
            'password' : self.secret['PASSWORD'],
        }    
    
    
    
    def _step_deploying(self):
        
        deploy_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value('DEPLOY_PROCESSING','framework_version'),
            role=self.execution_role,
            instance_type=self.args.config.get_value('DEPLOY_PROCESSING','instance_type'),
            instance_count=self.args.config.get_value('DEPLOY_PROCESSING','instance_count', dtype='int'),
            sagemaker_session=self.pipeline_session
        )
        
        step_deploy_args = deploy_processor.run(
            code='./create_endpoint.py',
            source_dir='./source/endpoint',
            git_config=self.git_config,
            inputs=None,
            outputs=None,
            arguments=[
                '--region', self.args.config.get_value("COMMON", "region"),
                '--tags', self.args.config.get_value("COMMON", "tags"),
                '--role', self.execution_role,
                '--lounge_name', self.args.config.get_value("COMMON", "lounge_name"),
                
                '--model_package_group_name', self.args.config.get_value("DEPLOYING", "model_package_group_name"),
                '--realtime_instance_type', self.args.config.get_value("DEPLOYING", "realtime_instance_type"),
                '--realtime_initial_instance_count', self.args.config.get_value("DEPLOYING", "realtime_initial_instance_count"),
            ]
        )
        
        self.deploying_process = ProcessingStep(
            name = "ModelDeployingProcess",
            step_args = step_deploy_args,
        )
        
        
        
    def _get_pipeline(self, ):
                
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[self.deploying_process],
            sagemaker_session=self.pipeline_session
        )
    
        return pipeline
    
    
        
    def execution(self, ):
        
        self._step_deploying()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.execution_role)
        execution = pipeline.start()
        desc = execution.describe()
        
        print(desc)
    
    
if __name__=="__main__":
    
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    config_file = "dev_deploy_config.ini"
    # config_file = "prd_deploy_config.ini"
    args.config = config_handler(config_file)
    
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe = ModelEndpointPipeline(args)
    pipe.execution()