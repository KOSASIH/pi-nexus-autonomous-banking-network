import requests

class DataFeedIntegration:
    def __init__(self, feed_url):
        self.feed_url = feed_url

    def fetch_feed(self):
        response = requests.get(self.feed_url)
        return response.json()

# Example usage
if __name__ == "__main__":
    data_feed = DataFeedIntegration('https://data-feed-url/api')
    feed_data = data_feed.fetch_feed()
    print(f"Fetched Feed Data: {feed_data}")
