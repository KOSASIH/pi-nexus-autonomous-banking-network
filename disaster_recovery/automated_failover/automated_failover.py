import azure.mgmt.recoveryservices as recoveryservices
import azure.mgmt.recoveryservices.models as recoverymodels
import time

# Set up Azure Recovery Services client
subscription_id = 'your_subscription_id'
resource_group_name = 'your_resource_group_name'
vault_name = 'your_vault_name'
recovery_services_client = recoveryservices.RecoveryServicesClient(credentials, subscription_id)

# Set up failover settings
fabric_name = 'your_fabric_name'
protection_container_name = 'your_protection_container_name'
replication_policy_name = 'your_replication_policy_name'
source_server_name = 'your_source_server_name'
target_server_name = 'your_target_server_name'

# Set up failover parameters
failover_parameters = recoverymodels.FailoverRequestParameters(
    failover_direction=recoverymodels.FailoverDirection.automatic,
    target_replication_policy_id=target_replication_policy_id,
    target_failover_location=target_failover_location
)

# Perform failover
failover_result = recovery_services_client.replication_fabrics.failover(
    fabric_name, protection_container_name, source_server_name, failover_parameters
)

# Wait for failover to complete
print("Failover initiated...")
while failover_result.properties.failover_operation_status != recoverymodels.FailoverOperationStatus.succeeded:
    time.sleep(10)
    failover_result = recovery_services_client.replication_fabrics.get_failover_status(
        fabric_name, protection_container_name, source_server_name
    )
print("Failover completed!")

# Perform failback
failback_parameters = recoverymodels.FailoverRequestParameters(
    failover_direction=recoverymodels.FailoverDirection.automatic,
    target_replication_policy_id=source_replication_policy_id,
    target_failover_location=source_failover_location
)

failback_result = recovery_services_client.replication_fabrics.failover(
    fabric_name, protection_container_name, target_server_name, failback_parameters
)

# Wait for failback to complete
print("Failback initiated...")
while failback_result.properties.failover_operation_status != recoverymodels.FailoverOperationStatus.succeeded:
    time.sleep(10)
    failback_result = recovery_services_client.replication_fabrics.get_failback_status(
        fabric_name, protection_container_name, target_server_name
    )
print("Failback completed!")
