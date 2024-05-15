# jira.py

import requests


def create_issue(summary, description, issuetype, project):
    url = "https://your-domain.atlassian.net/rest/api/3/issue/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Basic "
        + b64encode(("your-username:your-api-token").encode()).decode(),
    }
    data = {
        "fields": {
            "project": {"key": project},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issuetype},
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Issue created successfully")
    else:
        print("Error creating issue: ", response.json()["errors"])
