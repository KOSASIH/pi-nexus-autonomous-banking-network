import os

class CommunityDrivenDevelopment:
    def __init__(self):
        self.contributors = []

    def add_contributor(self, contributor):
        self.contributors.append(contributor)

    def submit_pull_request(self, contributor, pull_request):
        # Review and merge pull request
        if self.validate_pull_request(pull_request):
            self.merge_pull_request(pull_request)
            print(f"Pull request from {contributor} merged successfully!")
        else:
            print(f"Pull request from {contributor} rejected.")

    def validate_pull_request(self, pull_request):
        # Validate pull request
        return True

    def merge_pull_request(self, pull_request):
        # Merge pull request into main codebase
        os.system(f"git merge {pull_request}")

if __name__ == '__main__':
    cdd = CommunityDrivenDevelopment()
    contributor1 = 'Alice'
    contributor2 = 'Bob'
    cdd.add_contributor(contributor1)
    cdd.add_contributor(contributor2)
    pull_request1 = 'feature/new-feature'
    pull_request2 = 'fix/bug-fix'
    cdd.submit_pull_request(contributor1, pull_request1)
    cdd.submit_pull_request(contributor2, pull_request2)
