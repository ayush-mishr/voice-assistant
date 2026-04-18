import traceback
import asyncio
import boto3
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
from aws_sdk_bedrock_runtime.config import Config
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamOperationInput

async def main():
    try:
        session = boto3.Session(region_name='us-east-1')
        creds = session.get_credentials()
        frozen = creds.get_frozen_credentials() if creds else None
        print("Got creds:", frozen is not None)

        config = Config(
            region='us-east-1',
            aws_access_key_id=frozen.access_key if frozen else None,
            aws_secret_access_key=frozen.secret_key if frozen else None,
            aws_session_token=frozen.token if frozen else None
        )
        client = BedrockRuntimeClient(config=config)
        print('Client init OK')
        
        print('Invoking...')
        res = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id='amazon.nova-sonic-v1:0')
        )
        print('Invoke OK!')
    except Exception as e:
        print('Exception:', type(e))
        traceback.print_exc()

asyncio.run(main())
