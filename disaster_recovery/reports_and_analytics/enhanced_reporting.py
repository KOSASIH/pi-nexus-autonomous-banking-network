import pandas as pd
import plotly.express as px

# Load data
data = pd.read_csv("data.csv")

# Generate reports


def generate_activity_report():
    """Generate a report showing user activity."""
    activity = data[["user_id", "timestamp"]].groupby("user_id").count().reset_index()
    activity.columns = ["user_id", "activity_count"]
    return activity


def generate_usage_report():
    """Generate a report showing system usage."""
    usage = data[["system_id", "timestamp"]].groupby("system_id").count().reset_index()
    usage.columns = ["system_id", "usage_count"]
    return usage


def generate_error_report():
    """Generate a report showing system errors."""
    errors = data[data["error_code"] != 0]
    errors = (
        errors[["system_id", "error_code"]].groupby("system_id").count().reset_index()
    )
    errors.columns = ["system_id", "error_count"]
    return errors


# Generate visualizations


def generate_activity_chart():
    """Generate a chart showing user activity."""
    activity = generate_activity_report()
    fig = px.bar(activity, x="user_id", y="activity_count", title="User Activity")
    fig.show()


def generate_usage_chart():
    """Generate a chart showing system usage."""
    usage = generate_usage_report()
    fig = px.bar(usage, x="system_id", y="usage_count", title="System Usage")
    fig.show()


def generate_error_chart():
    """Generate a chart showing system errors."""
    errors = generate_error_report()
    fig = px.bar(errors, x="system_id", y="error_count", title="System Errors")
    fig.show()


# Run reports and visualizations
generate_activity_report()
generate_usage_report()
generate_error_report()
generate_activity_chart()
generate_usage_chart()
generate_error_chart()
