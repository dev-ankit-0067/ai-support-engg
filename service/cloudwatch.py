import boto3
from datetime import datetime, timedelta, timezone
from io import StringIO

session = boto3.Session(profile_name="default", region_name="us-east-1")
logs_client = session.client("logs", region_name="us-east-1")

log_group = "/aws-glue/jobs/error"

# start_time = int((datetime.now(timezone.utc) - timedelta(days=300)).timestamp() * 1000)
# end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

def get_log_details(job_id):
    buffer = StringIO()
    streams = logs_client.describe_log_streams(
        logGroupName=log_group,
        logStreamNamePrefix=job_id
    )["logStreams"]
    
    for s in streams:
        print(s)
        response = logs_client.get_log_events(
            logGroupName=log_group,
            logStreamName=s["logStreamName"],
            startFromHead=True
        )
        for event in response["events"]:
            buffer.write(event['message'])
            
    
    final_string = buffer.getvalue()
    buffer.close()
    
    with open("./data/spark_log.txt", "w") as f:
        f.write(final_string)


# get_log_details('')
