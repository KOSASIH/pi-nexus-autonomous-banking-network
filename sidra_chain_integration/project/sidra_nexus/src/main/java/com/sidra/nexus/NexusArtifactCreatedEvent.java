package com.sidra.nexus;

import org.springframework.context.ApplicationEvent;

public class NexusArtifactCreatedEvent extends ApplicationEvent {
    private NexusArtifact artifact;

    public NexusArtifactCreatedEvent(NexusArtifact artifact) {
        super(artifact);
        this.artifact = artifact;
    }

    public NexusArtifact getArtifact() {
        return artifact;
    }
}
