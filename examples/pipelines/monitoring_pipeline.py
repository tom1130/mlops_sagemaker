import os
import argparse
from pprint import pprint
from config.config import config_handler
from pytz import timezone
from datetime import datetime

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.pipeline_context import PipelineSession

from secret_manager.secret_manager import get_secret

class MonirotingPipeline:
    
    def __init__(self, args):

        self.args = args
        # git id, password 가져오기
        self.secret = get_secret()
        self._env_setting()

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

        self.excution_role = self.args.config.get_value("COMMON", "role")
        self.pipeline_name = self.args.config.get_value('COMMON','pipeline_name')
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

    def _step_monitoring(self):
        
        pipeline_session = PipelineSession()

        prefix_monitor = '/opt/ml/processing/'
        prediction_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'prediction_data_path')
        )
        label_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'label_data_path')
        )
        daily_label_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'labels_data_path')
        )
        output_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'monitoring_result_path')
        )

        monitoring_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("MONITORING", "framework_version"),
            role=self.excution_role,
            instance_type=self.args.config.get_value('MONITORING','instance_type'),
            instance_count=self.args.config.get_value('MONITORING','instance_count', dtype='int'),
            base_job_name='monitoring',
            sagemaker_session = pipeline_session,
            tags=self.tags
        )

        step_args = monitoring_processor.run(
            code='./monitoring.py',
            source_dir='./source/monitoring/',
            git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name='prediction-input',
                    source=prediction_path,
                    destination=os.path.join(prefix_monitor, 'input', 'predictions') 
                ),
                ProcessingInput(
                    input_name='label-input',
                    source=label_path,
                    destination=os.path.join(prefix_monitor, 'input', 'label')
                ),
                ProcessingInput(
                    input_name='labels-input',
                    source=daily_label_path,
                    destination=os.path.join(prefix_monitor, 'input', 'labels')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='output',
                    source=os.path.join(prefix_monitor, 'output', 'monitoring'),
                    destination=os.path.join(output_path, args.today)
                ),
                ProcessingOutput(
                    output_name='label-output',
                    source=os.path.join(prefix_monitor, 'output', 'label'),
                    destination=label_path
                )
            ],
            arguments=['--today', self.args.today],
            job_name = 'monitoring'
        )

        self.monitoring_process = ProcessingStep(
            name='MonitoringProcess',
            step_args=step_args,
            cache_config=self.cache_config
        )

        ## logging ################
        print("  \n== Monitoring processing Step ==")
        print("   \nArgs: ")
        for key, value in self.monitoring_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.monitoring_process.properties))

    def _get_pipeline(self):

        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[self.monitoring_process],
            sagemaker_session=self.pipeline_session
        )

        return pipeline

    def execution(self):
        
        self._step_monitoring()

        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.excution_role)
        execution = pipeline.start()

        print(execution.describe())
    
if __name__=='__main__':
    # change directory path
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    # get today date
    today = datetime.strftime(datetime.now(timezone('Asia/Seoul')), '%Y%m%d')

    # get config and argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', default=today)
    args, _ = parser.parse_known_args()

    # args.config = config_handler('prd_monitoring_config.ini')
    args.config = config_handler('dev_monitoring_config.ini')

    # execute monitoring pipeline
    pipe_monitor = MonirotingPipeline(args)
    pipe_monitor.execution()



    