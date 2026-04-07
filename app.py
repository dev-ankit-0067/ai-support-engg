import streamlit as st
import boto3
import pandas as pd
# from .service import cloudwatch
from ai_agents import ai_support_engginer

# -------------------------------
# AWS Glue Client
# -------------------------------
#  streamlit run .\streamlit_app.py --server.port 8502
session = boto3.Session(profile_name="default", region_name="us-east-1")
logs_client = session.client("logs", region_name="us-east-1")

@st.cache_resource
def get_glue_client():
    return session.client("glue", region_name="us-east-1")


# -------------------------------
# Fetch Glue Jobs
# -------------------------------
def get_glue_jobs(glue_client):
    jobs = []
    paginator = glue_client.get_paginator("get_jobs")

    for page in paginator.paginate():
        for job in page["Jobs"]:
            jobs.append(job["Name"])

    return sorted(jobs)


# -------------------------------
# Fetch Last 2 Runs for a Job
# -------------------------------
def get_last_two_runs(glue_client, job_name):
    response = glue_client.get_job_runs(
        JobName=job_name,
        MaxResults=2
    )

    runs = response.get("JobRuns", [])

    data = []
    for run in runs:
        data.append({
            "JobName": job_name,
            "JobRunId": run["Id"],
            "Status": run["JobRunState"],
            "StartedOn": run.get("StartedOn"),
            "CompletedOn": run.get("CompletedOn")
        })

    return pd.DataFrame(data)


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Glue Job Monitor", layout="wide")

st.title("🧩 AWS Glue Job Monitor")

glue_client = get_glue_client()

# -------------------------------
# Job Selection
# -------------------------------
job_list = get_glue_jobs(glue_client)

selected_job = st.selectbox(
    "Select Glue Job",
    job_list
)

# -------------------------------
# Show Last 2 Runs
# -------------------------------
if selected_job:
    df_runs = get_last_two_runs(glue_client, selected_job)

    if not df_runs.empty:
        st.subheader(f"Last 2 Runs for {selected_job}")

        st.dataframe(df_runs, use_container_width=True)

        # -------------------------------
        # Run Selection
        # -------------------------------
        selected_run = st.selectbox(
            "Select Run",
            df_runs["JobRunId"]
        )

        # -------------------------------
        # Display Selected Run ID
        # -------------------------------
        if selected_run:
            if df_runs.loc[df_runs["JobRunId"] == selected_run, "Status"].values[0] == 'FAILED':
                st.error(f" :x: Selected JobRunId: {selected_run}")
            else:
                st.success(f"✅ Selected JobRunId: {selected_run}")
        
        if "log_details" not in st.session_state:
            st.session_state.log_details = ""
          
        col1, col2, col3 = st.columns(3)
        
        with col1:
            error_logs = st.button(":open_book: Error Message")
          
        with col2:
            analyze_clicked = st.button("🔍 Analyze Failure")
        
        with col3:
            jira_clicked = st.button("🧾 Create Jira Ticket")
        
        # -------------------------------
        # Placeholder (Single Output Area)
        # -------------------------------
        output_placeholder = st.empty()
        
        if error_logs:
            output_placeholder.empty()  # Clear previous output
            with st.spinner(f"Fetching error logs : {selected_run}"):
                result = ai_support_engginer.get_rca("filtered_log",selected_run)
              
            st.session_state.log_details = result
          
        if analyze_clicked:
            output_placeholder.empty()
            with st.spinner(f"Analyzing failure : {selected_run}"):
                result = ai_support_engginer.get_rca("rca_report",selected_run)
            st.session_state.log_details = result
        
        if jira_clicked:
            output_placeholder.empty()
            with st.spinner(f"Creating Jira ticket for: {selected_run}"):
                result = ai_support_engginer.get_rca("jira_ticket", selected_run)
            
            st.session_state.log_details = result
        
        # -------------------------------
        # Render Latest Output Only
        # -------------------------------
        if st.session_state.log_details:
            output_placeholder.markdown(st.session_state.log_details)

    else:
        st.warning("No runs found for this job")