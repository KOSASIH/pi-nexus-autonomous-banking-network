import networkx as nx

class TrustBasedSecurity:
    def __init__(self):
        self.social_graph = nx.Graph()

    def add_user(self, user):
        self.social_graph.add_node(user)

    def add_friendship(self, user1, user2):
        self.social_graph.add_edge(user1, user2)

    def calculate_trust_score(self, user):
        # Calculate trust score based on social connections
        trust_score = 0
        for neighbor in self.social_graph.neighbors(user):
            trust_score += self.social_graph.degree(neighbor)
        return trust_score

if __name__ == '__main__':
    tbs = TrustBasedSecurity()
    user1 = 'Alice'
    user2 = 'Bob'
    user3 = 'Charlie'
    tbs.add_user(user1)
    tbs.add_user(user2)
    tbs.add_user(user3)
    tbs.add_friendship(user1, user2)
    tbs.add_friendship(user2, user3)
    print(tbs.calculate_trust_score(user1))  # 2
    print(tbs.calculate_trust_score(user2))  # 3
    print(tbs.calculate_trust_score(user3))  # 2
