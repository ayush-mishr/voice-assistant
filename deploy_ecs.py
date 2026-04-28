import boto3
import os
from dotenv import load_dotenv

load_dotenv(os.path.join("backend", ".env"))

ecs = boto3.client('ecs', region_name='us-east-1')
cluster = 'voice-assistant-cluster'
service = 'voice-assistant-backend-svc'

print(f"Forcing new deployment for {service}...")
ecs.update_service(
    cluster=cluster,
    service=service,
    forceNewDeployment=True
)

print(f"Finding running tasks for {service}...")
tasks = ecs.list_tasks(
    cluster=cluster,
    serviceName=service
)['taskArns']

if tasks:
    print(f"Found {len(tasks)} tasks. Stopping them to force immediate recreation...")
    for task_arn in tasks:
        ecs.stop_task(cluster=cluster, task=task_arn)
        print(f"Stopped {task_arn}")
else:
    print("No running tasks found.")

print("Done. ECS will now launch new tasks with the latest image.")
