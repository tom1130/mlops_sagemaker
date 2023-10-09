import boto3


def get_cloudwatch_url(region_name, pipeline_name, status):
    sagemaker_client = boto3.client("sagemaker", region_name=region_name)
    
    pipeline_executions = sagemaker_client.list_pipeline_executions(
        PipelineName=pipeline_name,
        SortBy='CreationTime',
        SortOrder='Descending',
    )['PipelineExecutionSummaries']

    for exec in pipeline_executions:
        if exec['PipelineExecutionStatus']==status:
            exec_id = exec['PipelineExecutionArn'].split('/')[-1]
            break

    cloudwatch_url = f"https://{region_name}.console.aws.amazon.com/cloudwatch/home?region={region_name}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FProcessingJobs$3FlogStreamNameFilter$3Dpipelines-{exec_id}"
    return cloudwatch_url



def publish_sns(region_name, sns_arn, project_name, pipeline_name, error_type, error_message, env, status="Executing"):
    sns_client = boto3.client("sns", region_name=region_name)
    error_message = str(error_message)
    cloudwatch_url = get_cloudwatch_url(region_name, pipeline_name, status)
    subject = "[AWS] {}({}) Error Notice".format(project_name, env.upper())
    message = '''
    다음과 같이 에러가 발생하였으니 조치하여 주시기 바랍니다.

    1. Project 명 : {}

    2. Pipeline : {}

    3. Error 종류 : {}

    4. Error message : {}

    5. Cloud watch Log : {}
    '''.format(project_name, pipeline_name, error_type, error_message, cloudwatch_url)
    
    response = sns_client.publish(
        TargetArn=sns_arn,
        Message=message,
        Subject=subject,
    )