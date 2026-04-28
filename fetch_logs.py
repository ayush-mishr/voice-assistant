import boto3
import time
import os
from dotenv import load_dotenv

load_dotenv(os.path.join("backend", ".env"))

client = boto3.client('logs', region_name='us-east-1')
log_group = '/ecs/voice-assistant-backend'

# Get the most recent log streams
streams = client.describe_log_streams(
    logGroupName=log_group,
    orderBy='LastEventTime',
    descending=True,
    limit=3
)

if not streams['logStreams']:
    print("No streams found")
    exit()

with open('ecs_debug_logs.txt', 'w', encoding='utf-8') as f:
    for stream in streams['logStreams']:
        stream_name = stream['logStreamName']
        f.write(f"\n{'='*80}\n")
        f.write(f"STREAM: {stream_name}\n")
        f.write(f"{'='*80}\n\n")
        
        events = client.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            limit=100,
            startFromHead=False
        )
        
        for e in events['events']:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(e['timestamp']/1000))
            f.write(f"[{ts}] {e['message']}\n")

print("Logs written to ecs_debug_logs.txt")
