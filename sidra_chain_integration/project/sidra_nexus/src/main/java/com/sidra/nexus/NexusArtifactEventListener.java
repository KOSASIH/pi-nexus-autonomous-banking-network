package com.sidra.nexus;

import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

@Component
public class NexusArtifactEventListener {
    @EventListener
    public void handleArtifactCreatedEvent(NexusArtifactCreatedEvent event) {
        // Handle artifact created event
    }

    @EventListener
    public void handleArtifactUpdatedEvent(NexusArtifactUpdatedEvent event) {
        // Handle artifact updated event
    }

    @EventListener
    public void handleArtifactDeletedEvent(NexusArtifactDeletedEvent event) {
        // Handle artifact deleted event
    }
}
