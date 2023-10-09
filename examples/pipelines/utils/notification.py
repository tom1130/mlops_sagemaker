import boto3
from urllib import parse

from utils.secret_manager import get_secret

def check_error_text(error, env):
    secret = get_secret(env)

    username = secret['USER'].split('@')[0]
    password = secret['PASSWORD']

    confirm_text = parse.quote(username)+':'+parse.quote(password)

    if confirm_text in error:
        error = error.replace(confirm_text, '')
    
    return error

def publish_sns(region_name, sns_arn, project_name, pipeline_name, error_type, error_message, env):
    error_message = str(error_message)
    sns_client = boto3.client("sns", region_name=region_name)
    subject = "[AWS] {}({}) Error Notice".format(project_name, env.upper())

    error_message = check_error_text(error_message, env)

    message = '''
    다음과 같이 에러가 발생하였으니 조치하여 주시기 바랍니다.

    1. Project 명 : {}

    2. Pipeline : {}

    3. Error 종류 : {}

    4. Error message : {}
    '''.format(project_name, pipeline_name, error_type, error_message)

    response = sns_client.publish(
        TargetArn=sns_arn,
        Message=message,
        Subject=subject,
    )

    return response