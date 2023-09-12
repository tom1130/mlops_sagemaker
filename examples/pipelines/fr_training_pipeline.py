import os
import argparse
from datetime import datetime
from pytz import timezone
from config.config import config_handler
from pprint import pprint

from sagemaker.sklearn import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor

from secret_manager.secret_manager import get_secret


class FrTrainingPipeline():
    '''
    boto3 활용 pipeline
    '''
    def __init__(self, args):
        
        self.args = args
        self.secret = get_secret()
        
        self._env_setting()
        
        
    def _env_setting(self, ):
        
        self.pipeline_session = PipelineSession()
        self.execution_role = self.args.config.get_value("COMMON", "role")
        self.pipeline_name = self.args.config.get_value("COMMON", "pipeline_name")
        self.tags = self.args.config.get_value('COMMON', 'tags', dtype='list') 
        self.today = self.args.today
        self.now = datetime.now(timezone("Asia/Seoul")).strftime("%H%M%S")
        
        self.git_config = {
            'repo' : self.args.config.get_value('GIT','git_repo'),
            'branch' : self.args.config.get_value('GIT','git_branch'),
            'username' : self.secret['USER'].split('@')[0],
            'password' : self.secret['PASSWORD'],
        }

        self.data_bucket = self.args.config.get_value('COMMON','data_bucket')
        self.code_bucket = self.args.config.get_value('COMMON','code_bucket')
        self.prefix = self.args.config.get_value('COMMON','prefix')
        # s3://
        self.base_path = os.path.join(
            's3://',
            self.data_bucket,
            self.prefix
        )
        

        
    def _step_preprocessing(self, ):
        
        prefix_prep = '/opt/ml/processing'
        data_path = os.path.join(
            self.base_path, 
            self.args.config.get_value('PREPROCESSING','data_path')
        )
        target_path = os.path.join(
            self.base_path, 
            self.args.config.get_value('PREPROCESSING','target_path')
        )
        etc_path = os.path.join(
            self.base_path, 
            self.args.config.get_value('PREPROCESSING','etc_path')
        )
        
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("PREPROCESSING", "framework_version"),
            role=self.execution_role,
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count", dtype='int'),
            sagemaker_session=self.pipeline_session,
            tags=self.tags
        )
        
        step_preprocessing_args = prep_processor.run(
            code = './fr_training_preprocess.py',
            source_dir = './source/preprocess',
            git_config=self.git_config,
            inputs = [
                ProcessingInput(
                    input_name='input',
                    source=data_path,
                    destination=os.path.join(prefix_prep, 'input')
                ),
                ProcessingInput(
                    input_name='etc',
                    source=etc_path,
                    destination=os.path.join(prefix_prep, 'etc')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train-data",
                    source=os.path.join(prefix_prep, "output", "train"),
                    destination=os.path.join(target_path,'fr','train-data'),
                ),
                ProcessingOutput(
                    output_name="validation-data",
                    source=os.path.join(prefix_prep, "output", "validation"),
                    destination=os.path.join(target_path,'fr','validation-data'),
                ),
                ProcessingOutput(
                    output_name="test-data",
                    source=os.path.join(prefix_prep, "output", "test"),
                    destination=os.path.join(target_path,'fr','test-data'),
                )
            ]
        )
        
        self.preprocessing_process = ProcessingStep(
            name = "FrTrainPreprocessingProcess",
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

        
    def _step_training(self, ):
        
        train_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("TRAIN_PROCESSING", "framework_version"), #
            role=self.execution_role,
            instance_type=self.args.config.get_value("TRAIN_PROCESSING", "instance_type"), #
            instance_count=self.args.config.get_value("TRAIN_PROCESSING", "instance_count", dtype="int"), #
            sagemaker_session=self.pipeline_session,
        )

        step_train_processing_args = train_processor.run(
            code='./train.py',
            source_dir="./source/train",
            git_config=self.git_config,
            inputs=None,
            outputs=None,
            arguments=[
                '--today', self.today,
                '--now', self.now,
                '--role', self.execution_role,
                '--data_bucket', self.data_bucket,
                '--code_bucket', self.code_bucket,
                '--prefix', self.prefix,
                '--lounge_name', self.args.config.get_value("COMMON", "lounge_name"),
                '--region', self.args.config.get_value("COMMON", "region"),
                '--tags', self.args.config.get_value("COMMON", "tags"),
                
                '--auto_ml_algorithms', self.args.config.get_value("TRAIN_SCRIPT", "auto_ml_algorithms"),
                '--max_candidate', self.args.config.get_value("TRAIN_SCRIPT", "max_candidate"),
                '--max_runtime_per_training_job', self.args.config.get_value("TRAIN_SCRIPT", "max_runtime_per_training_job"),
                '--max_runtime_for_auto_ml_job', self.args.config.get_value("TRAIN_SCRIPT", "max_runtime_for_auto_ml_job"),
                '--mode', self.args.config.get_value("TRAIN_SCRIPT", "mode"),
                '--target_attribute', self.args.config.get_value("TRAIN_SCRIPT", "target_attribute"),
                '--job_objective', self.args.config.get_value("TRAIN_SCRIPT", "job_objective"),

                '--transform_instance_type', self.args.config.get_value("TRAIN_SCRIPT", "transform_instance_type"),
                '--transform_instance_count', self.args.config.get_value("TRAIN_SCRIPT", "transform_instance_count"),

                '--registry_content_type', self.args.config.get_value("TRAIN_SCRIPT", "registry_content_type"),
                '--registry_response_type', self.args.config.get_value("TRAIN_SCRIPT", "registry_response_type"),
                '--registry_realtime_instance_type', self.args.config.get_value("TRAIN_SCRIPT", "registry_realtime_instance_type"),
                '--registry_transform_instance_type', self.args.config.get_value("TRAIN_SCRIPT", "registry_transform_instance_type"),
                '--model_package_group_name', self.args.config.get_value("COMMON", "model_package_group_name"),
            ]
        )


        self.training_process = ProcessingStep(
            name = "FrTrainingProcess",
            step_args = step_train_processing_args,
        )
        
        self.training_process.add_depends_on([self.preprocessing_process])

        ## logging ##########
        print("  \n== Training Step ==")
        print("   \nArgs: ")
        for key, value in self.training_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.training_process.properties))
        
        
    def _get_pipeline(self, ):
                
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[
                self.preprocessing_process, 
                self.training_process, 
            ],
            sagemaker_session=self.pipeline_session
        )
    
        return pipeline
    
        
    def execution(self, ):
        
        self._step_preprocessing()
        self._step_training()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.execution_role)
        execution = pipeline.start()
        desc = execution.describe()
        
        print(desc)
    
    
if __name__=="__main__":
    
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    
    today = datetime.strftime(datetime.now(timezone('Asia/Seoul')), '%Y%m%d')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', default=today)
    args, _ = parser.parse_known_args()

    config_file = "dev_fr_train_config.ini"
    # config_file = "prd_fr_train_config.ini"
    args.config = config_handler(config_file)
    
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe = FrTrainingPipeline(args)
    pipe.execution()