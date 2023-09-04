import os
import argparse
from config.config import config_handler

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.pipeline_context import PipelineSession

class pipeline_monitoring:
    
    def __init__(self, args):

        self.args = args

        self._env_setting()

    def _env_setting(self):

        self.strExcutionRole = self.args.config.get_value("COMMON", "role")
        self.strBucketName = self.args.config.get_value("COMMON", "bucket")
        self.strPipelineName = self.args.config.get_value('COMMON','pipeline_name')
        
        self.cache_config = CacheConfig(
            enable_caching=self.args.config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.args.config.get_value("PIPELINE", "expire_after")
        )

        self.git_config = {
            'repo' : self.args.config.get_value('GIT','git_repo'),
            'branch' : 'master',
            'username' : self.args.config.get_value('GIT','git_user'),
            'password' : self.args.config.get_value('GIT','git_password'),
        }

        self.pipeline_session = PipelineSession()

    def _step_monitoring(self):
        
        pipeline_session = PipelineSession()

        strPrefixMonitor = '/opt/ml/processing/'
        strPredictionPath = self.args.config.get_value('MONITORING', 'prediction_data_path')
        strLabelPath = self.args.config.get_value('MONITORING', 'label_data_path')
        strOutputPath = self.args.config.get_value('MONITORING', 'monitoring_result_path')

        monitoring_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("MONITORING", "framework_version"),
            role=self.strExcutionRole,
            instance_type=self.args.config.get_value('MONITORING','instance_type'),
            instance_count=self.args.config.get_value('MONITORING','instance_count', dtype='int'),
            base_job_name='monitoring',
            sagemaker_session = pipeline_session
        )

        step_args = monitoring_processor.run(
            code='./monitoring.py',
            source_dir='./source/monitoring/',
            git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name='prediction-input',
                    source=strPredictionPath,
                    destination=os.path.join(strPrefixMonitor, 'input', 'predictions') 
                ),
                ProcessingInput(
                    input_name='label-input',
                    source=strLabelPath,
                    destination=os.path.join(strPrefixMonitor, 'input', 'label')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='output',
                    source=os.path.join(strPrefixMonitor, 'output'),
                    destination=os.path.join(strOutputPath, args.today)
                )
            ],
            # arguments=['--prefix_prep', strPrefixPrep],
            job_name = 'monitoring'
        )

        self.monitoring_process = ProcessingStep(
            name='MonitoringProcess',
            step_args=step_args,
            cache_config=self.cache_config
        )

        ## logging ################

    def _get_pipeline(self):

        pipeline = Pipeline(
            name=self.strPipelineName,
            steps=[self.monitoring_process],
            sagemaker_session=self.pipeline_session
        )

        return pipeline

    def execution(self):
        
        self._step_monitoring()

        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.strExcutionRole)
        execution = pipeline.start()

        print(execution.describe())
    
if __name__=='__main__':
    # change directory path
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    # get config and argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', default='20230725')
    args, _ = parser.parse_known_args()
    args.config = config_handler('monitoring_config.ini')

    # execute monitoring pipeline
    pipe_monitor = pipeline_monitoring(args)
    pipe_monitor.execution()



    