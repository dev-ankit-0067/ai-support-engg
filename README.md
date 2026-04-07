# AI Support Engineer

A Streamlit-based AWS Glue job monitoring interface that retrieves job run history and provides failure analysis using an LLM-powered support agent pipeline.

## Overview

This project is designed to help support engineers investigate AWS Glue job failures by:

- Listing AWS Glue jobs
- Showing the latest two runs for a selected job
- Fetching Glue error logs from CloudWatch
- Summarizing log chunks and identifying likely root causes
- Generating both an RCA report and a JIRA-style ticket description

## Key Components

- `app.py`
  - Streamlit UI for job selection, run selection, and action buttons
  - Uses `boto3` to query Glue jobs and job runs
  - Calls the AI agent pipeline for log analysis and ticket generation

- `ai_agents/ai_support_engginer.py`
  - Defines log filtering, chunking, and LLM prompt flows
  - Uses LangGraph and LangChain to build a multi-agent workflow
  - Includes summarization, error extraction, RCA, and Jira ticket generation

- `service/cloudwatch.py`
  - Retrieves CloudWatch log events from the `/aws-glue/jobs/error` log group
  - Writes logs to `./data/spark_log.txt`

## Requirements

Dependencies are listed in `requirement.txt`.

Important libraries:

- `streamlit`
- `boto3`
- `pandas`
- `python-dotenv`
- `langchain`, `langgraph`, `langchain-huggingface`
- `huggingface_hub`

## Setup

1. Clone the repository and change into the project directory.
2. Create a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirement.txt
```

4. Create a `.env` file in the project root with the following variables:

```bash
HUGGINGFACEHUB_API_TOKEN=<your_huggingface_token>
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
```

5. Ensure your AWS CLI credentials and profile are configured for `default` in `~/.aws/credentials`.
   - `app.py` and `service/cloudwatch.py` both use `boto3.Session(profile_name="default", region_name="us-east-1")`

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the displayed local URL in your browser.

## Usage

1. Select an AWS Glue job from the dropdown.
2. Review the last two Glue job runs.
3. Choose a run from the second dropdown.
4. Click one of the actions:
   - `Error Message` to fetch logs
   - `Analyze Failure` to generate an RCA report
   - `Create Jira Ticket` to generate a Jira-style ticket draft

## Notes

- The app reads CloudWatch logs for Glue failures from `/aws-glue/jobs/error`.
- The `jira_ticket` feature currently generates ticket text only; it does not connect to a real Jira system.
- `ai_support_engginer.py` uses a Hugging Face endpoint and requires a valid API token.
- If your log file is large, the pipeline filters and chunks logs before sending to the model.

## File Structure

- `app.py` - Streamlit interface
- `ai_agents/ai_support_engginer.py` - Agent workflow for root cause analysis
- `service/cloudwatch.py` - CloudWatch log retrieval helper
- `data/spark_log.txt` - Generated local file for fetched logs
- `requirement.txt` - Python dependencies

## Troubleshooting

- If `HUGGINGFACEHUB_API_TOKEN` is missing, the app will raise a runtime error.
- Ensure the AWS `default` profile has access to Glue and CloudWatch logs.
- For region-specific Glue jobs or logs, update `region_name="us-east-1"` as needed.
