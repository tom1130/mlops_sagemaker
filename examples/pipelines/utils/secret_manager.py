import boto3
import json
from botocore.exceptions import ClientError

def get_secret(env):

    if env=='dev':
        secret_name = "arn:aws:secretsmanager:ap-northeast-2:993398491107:secret:git/dlk/prd/ml/bitbucket-Gz2PUy"
    elif env=='prd':
        secret_name = 'arn:aws:secretsmanager:ap-northeast-2:993398491107:secret:git/dlk/prd/ml/bitbucket-Gz2PUy'
    region_name = "ap-northeast-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']

    # Your code goes here.
    secret = json.loads(secret)
    return secret

if __name__=='__main__':
    pass