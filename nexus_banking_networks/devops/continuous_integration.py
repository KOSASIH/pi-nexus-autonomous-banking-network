import Jenkins

class ContinuousIntegration:
    def __init__(self, jenkins_url, job_name):
        self.jenkins_url = jenkins_url
        self.job_name = job_name
        self.jenkins_client = Jenkins.Jenkins(self.jenkins_url)

    def trigger_build(self):
        # Trigger Jenkins build
        self.jenkins_client.build_job(self.job_name)
