@Component
public class NexusArtifactEventListener {
    @EventListener
    public void handleArtifactCreatedEvent(NexusArtifactCreatedEvent event) {
        NexusArtifact artifact = event.getArtifact();
        // Send a notification to users, or trigger some other action
        System.out.println("New artifact created: " + artifact.getName());
    }
}
