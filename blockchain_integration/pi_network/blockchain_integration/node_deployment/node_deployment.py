import docker

def deploy_node(node_id, stellar_sdk_url):
    client = docker.from_env()
    container = client.containers.run(
        "pi-network-node",
        detach=True,
        environment={
            "NODE_ID": node_id,
            "STELLAR_SDK_URL": stellar_sdk_url
        }
    )
    return container
