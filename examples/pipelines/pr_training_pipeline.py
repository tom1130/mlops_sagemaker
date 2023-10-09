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
from sagemaker.workflow.parameters import ParameterString

from utils.secret_manager import get_secret
from utils.notification import publish_sns


class PrTrainingPipeline():
    '''
    boto3 활용 pipeline
    '''
    def __init__(self, args):
        # config 변수 가져오기
        self.args = args
        # git id, pw 가져오기
        self.secret = get_secret(args.env)
        # 초기 세팅
        self._env_setting()   
        
        
    def _env_setting(self, ):
        # monitoring.py내에서 pipeline trigger 시, 입력해야 하는 Parameter들을 ParameterString으로 정의
        self.today = ParameterString(
            name="today",
            default_value=datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d"),
        )

        self.now = ParameterString(
            name="now",
            default_value=datetime.now(timezone("Asia/Seoul")).strftime("%H%M%S")
        )
        
        # model register 시, status가 셋 중 하나가 아니면 오류 발생
        # pipeline 실행 단계에서 확인
        model_approval_status = self.args.model_approval_status
        if model_approval_status not in ("Approved", "PendingManualApproval", "Rejected"):
            raise ValueError("model_approval_status는 'Approved', 'PendingManualApproval', 'Rejected' 중 하나의 값을 가져야 합니다.")

        self.model_approval_status = ParameterString(
            name="model_approval_status",
            default_value=model_approval_status,
        )

        # 고정 변수 설정
        self.pipeline_session = PipelineSession()
        self.execution_role = self.args.config.get_value("COMMON", "role")
        self.sns_arn = self.args.config.get_value("SNS", "arn")
        self.pipeline_name = self.args.config.get_value("COMMON", "pipeline_name")
        self.tags = self.args.config.get_value('COMMON', 'tags', dtype='list') 
         
        self.git_config = {
            'repo' : self.args.config.get_value('GIT','git_repo'),
            'branch' : self.args.config.get_value('GIT','git_branch'),
            'username' : self.secret['USER'].split('@')[0],
            'password' : self.secret['PASSWORD'],
        }
                
        self.data_bucket = self.args.config.get_value('COMMON','data_bucket')
        self.code_bucket = self.args.config.get_value('COMMON','code_bucket')
        self.prefix = self.args.config.get_value('COMMON','prefix')

        # s3://wsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE
        self.base_path = os.path.join(
            's3://',
            self.data_bucket,
            self.prefix
        )



    def _step_preprocessing(self, ):
        # 경로 정의
        prefix_prep = '/opt/ml/processing'
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/train/raw
        data_path = os.path.join(
            self.base_path, 
            self.args.config.get_value('PREPROCESSING','data_path')
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/train/preprocess
        target_path = os.path.join(
            self.base_path, 
            self.args.config.get_value('PREPROCESSING','target_path')
        )
        # s3://awsdc-s3-dlk-dev(prd)-ml-data/ML-2023-P01-LOUNGE/etc
        etc_path = os.path.join(
            self.base_path, 
            self.args.config.get_value('PREPROCESSING','etc_path')
        )
        
        # FrameworkProcessor 정의 (Preprocessing)
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("PREPROCESSING", "framework_version"),
            role=self.execution_role,
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count", dtype='int'),
            sagemaker_session=self.pipeline_session,
            tags=self.tags
        )
        
        # Step Argument 정의
        # input : training용 raw data, holiday data
        # output : train, validation, test data (splitted)
        step_preprocessing_args = prep_processor.run(
            code = './pr_training_preprocess.py',
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
                    destination=os.path.join(target_path,'pr','train-data'),
                ),
                ProcessingOutput(
                    output_name="validation-data",
                    source=os.path.join(prefix_prep, "output", "validation"),
                    destination=os.path.join(target_path,'pr','validation-data'),
                ),
                ProcessingOutput(
                    output_name="test-data",
                    source=os.path.join(prefix_prep, "output", "test"),
                    destination=os.path.join(target_path,'pr','test-data'),
                )
            ],
            arguments=[
                '--region', self.args.config.get_value("COMMON", "region"),
                '--sns_arn', self.args.config.get_value("SNS", "arn"),
                '--project_name', self.args.config.get_value("COMMON", "project_name"),
                '--pipeline_name', self.args.config.get_value("COMMON", "pipeline_name"),
                '--env', self.args.config.get_value("COMMON", "env"),
            ]
        )
        
        # step 정의
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
        
        
    def _step_training(self, ):
        # FrameworkProcessor 정의 (Training AutoML, Evaluation, Registering)
        train_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("TRAIN_PROCESSING", "framework_version"), #
            role=self.execution_role,
            instance_type=self.args.config.get_value("TRAIN_PROCESSING", "instance_type"), #
            instance_count=self.args.config.get_value("TRAIN_PROCESSING", "instance_count", dtype="int"), #
            sagemaker_session=self.pipeline_session,
            tags=self.tags,
        )

        # Step argument 정의
        # input/output : None. train_processor 내의 스크립트가 S3 Uri로 직접 access함
        # arguments : training, batch transform, evaluation, registering에 필요한 arguments 전달함
        step_train_processing_args = train_processor.run(
            code='./train.py',
            source_dir="./source/train",
            git_config=self.git_config,
            inputs=None,
            outputs=None,
            arguments=[
                '--env', self.args.config.get_value("COMMON", "env"),
                '--today', self.today,
                '--now', self.now,
                '--role', self.execution_role,
                '--data_bucket', self.data_bucket,
                '--code_bucket', self.code_bucket,
                '--prefix', self.prefix,
                '--lounge_name', self.args.config.get_value("COMMON", "lounge_name"),
                '--region', self.args.config.get_value("COMMON", "region"),
                '--tags', self.args.config.get_value("COMMON", "tags"),
                '--sns_arn', self.args.config.get_value("SNS", "arn"),
                '--project_name', self.args.config.get_value("COMMON", "project_name"),
                '--pipeline_name', self.args.config.get_value("COMMON", "pipeline_name"),
                
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
                '--model_approval_status', self.model_approval_status,
            ]
        )

        # step 정의
        self.training_process = ProcessingStep(
            name = "PrTrainingProcess",
            step_args = step_train_processing_args,
        )
        
        # Preprocessing Step과 Training-Evaluation-Registering Step의 Dependancy 지정
        # 각 Step 간 input/output이 properties로 연결되어 있지 않으므로, 명시적으로
        # preprocessing_process가 끝난 이후 training_process를 시작하도록 함
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
        # pipeline 클래스 정의
        # pipeline명, 활용 step, session, 변수 정의        
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[
                self.preprocessing_process, 
                self.training_process, 
            ],
            parameters=[self.today, self.now, self.model_approval_status],
            sagemaker_session=self.pipeline_session
        )
    
        return pipeline
    
        
    def execution(self, ):
        # step을 실행하여 각각의 step 정의
        self._step_preprocessing()
        self._step_training()
        # pipeline 클래스 정의
        pipeline = self._get_pipeline()
        # pipeline upsert
        pipeline.upsert(role_arn=self.execution_role, tags=self.tags)
        # pipeline 실행
        execution = pipeline.start()
        desc = execution.describe()
        
        print(desc)
        execution.wait(max_attempts=120, delay=60)

        print("\n#####Execution completed. Execution step details:")
        print(execution.list_steps())
            
if __name__=="__main__":

    try:
        # 파일 실행위치 변경
        strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
        os.chdir(strBasePath)
        # get config and argument
        parser = argparse.ArgumentParser()
        # 환경 정의(dev, prd)
        parser.add_argument('--env', default='dev')
        # 등록될 모델의 ApprovalStatus를 정의. 디폴트값은 Approved이며, 스크립트 직접 실행 시 argument 활용하여 변경 가능
        parser.add_argument('--model_approval_status', default="Approved")
        args, _ = parser.parse_known_args()

        if args.env=='dev':
            config_file = "dev_pr_train_config.ini"
        elif args.env=='prd':
            config_file = "prd_pr_train_config.ini"
        args.config = config_handler(config_file)
        
        print("Received arguments {}".format(args))
        os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
        
        pipe = PrTrainingPipeline(args)
        pipe.execution()

    except Exception as e:

        publish_sns(region_name=args.config.get_value('COMMON','region'),
                    sns_arn=args.config.get_value('SNS','arn'),
                    project_name=args.config.get_value('COMMON','project_name'),
                    pipeline_name=args.config.get_value('COMMON','pipeline_name'),
                    error_type="PR training pipeline 실행 중 에러",
                    error_message=e,
                    env=args.config.get_value('COMMON','env')
                    )
        
        raise Exception('pipeline error')