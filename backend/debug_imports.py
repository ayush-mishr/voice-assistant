import sys
print('1. Loading dotenv')
from dotenv import load_dotenv
print('2. Loading os, json, uuid')
import os, json, uuid
print('3. Loading fastapi')
from fastapi import FastAPI
print('4. Loading auth')
import auth
print('5. Loading memory_short_term')
import memory_short_term
print('6. Loading memory_long_term')
import memory_long_term
print('7. Loading document_processor')
import document_processor
print('8. Loading agent_core')
import agent_core
print('9. Loading boto3')
import boto3
print('10. Loading bedrock runtime')
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
print('Done!')
