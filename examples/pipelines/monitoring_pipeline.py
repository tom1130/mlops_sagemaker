import os
import argparse
from pprint import pprint
from config.config import config_handler

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables

from utils.secret_manager import get_secret
from utils.notification import publish_sns

class MonirotingPipeline:
    
    def __init__(self, args):
        # config 변수 가져오기
        self.args = args
        # git id, password 가져오기
        self.secret = get_secret(args.env)
        self._env_setting()

    def _env_setting(self):
        # 변수 설정
        self.now = ExecutionVariables.START_DATETIME
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

        prefix_monitor = '/opt/ml/processing/'
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/transformed_output
        prediction_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'prediction_data_path')
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/train/raw
        label_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'label_data_path')
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/warehouse/lounge
        daily_label_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'labels_data_path')
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/monitoring
        output_path = os.path.join(
            self.base_path,
            self.args.config.get_value('MONITORING', 'monitoring_result_path')
        )
        # FrameworkProcessor 정의
        monitoring_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("MONITORING", "framework_version"),
            role=self.excution_role,
            instance_type=self.args.config.get_value('MONITORING','instance_type'),
            instance_count=self.args.config.get_value('MONITORING','instance_count', dtype='int'),
            base_job_name='monitoring',
            sagemaker_session = self.pipeline_session,
            tags=self.tags
        )
        # step argument 정의
        # input : prediction data, label data, daily label data
        # output : label data, monitoring result
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
                ),
                ProcessingInput(
                    input_name='monitoring-input',
                    source=output_path,
                    destination=os.path.join(prefix_monitor, 'input', 'monitoring')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='output',
                    source=os.path.join(prefix_monitor, 'output', 'monitoring'),
                    destination=output_path
                ),
                ProcessingOutput(
                    output_name='label-output',
                    source=os.path.join(prefix_monitor, 'output', 'label'),
                    destination=label_path
                )
            ],
            arguments=[
                '--sns_arn', self.args.config.get_value("SNS", "arn"),
                '--project_name', self.args.config.get_value("COMMON", "project_name"),
                '--env', self.args.config.get_value("COMMON", "env"),
                '--now', self.now,
                '--pipeline_name', self.args.config.get_value("COMMON", "pipeline_name"),
                '--fr_pipieline_name', self.args.config.get_value('MONITORING','fr_pipeline_name'),
                '--mr_pipieline_name', self.args.config.get_value('MONITORING','mr_pipeline_name'),
                '--pr_pipieline_name', self.args.config.get_value('MONITORING','pr_pipeline_name'),
            ],
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
        # pipeline 클래스 정의
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[self.monitoring_process],
            sagemaker_session=self.pipeline_session
        )

        return pipeline

    def execution(self):
        # monitoring step 실행
        self._step_monitoring()
        # pipeline 정의
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.excution_role, tags=self.tags)
        # pipeline 실행
        execution = pipeline.start()

        print(execution.describe())
        execution.wait(max_attempts=120, delay=60)

        print("\n#####Execution completed. Execution step details:")
        print(execution.list_steps())
    
if __name__=='__main__':
    try:
        # change directory path
        strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
        os.chdir(strBasePath)

        # get config and argument
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', default='dev')
        args, _ = parser.parse_known_args()

        if args.env=='dev':
            config_file = "dev_monitoring_config.ini"
        elif args.env=='prd':
            config_file = "prd_monitoring_config.ini"
        args.config = config_handler(config_file)

        # execute monitoring pipeline
        pipe_monitor = MonirotingPipeline(args)
        pipe_monitor.execution()

    except Exception as e:
        # 에러 시, sns 알림 
        publish_sns(region_name=args.config.get_value('COMMON','region'),
                    sns_arn=args.config.get_value('SNS','arn'),
                    project_name=args.config.get_value('COMMON','project_name'),
                    pipeline_name=args.config.get_value('COMMON','pipeline_name'),
                    error_type="Monitoring pipeline 실행 중 에러",
                    error_message=e,
                    env=args.config.get_value('COMMON','env')
                    )
        
        raise Exception('pipeline error')