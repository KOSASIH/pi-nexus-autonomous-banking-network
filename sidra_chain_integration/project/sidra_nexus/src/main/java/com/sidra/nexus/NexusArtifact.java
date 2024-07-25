package com.sidra.nexus;

import lombok.Data;

@Data
public class NexusArtifact {
    private Long id;
    private String name;
    private String groupId;
    private String artifactId;
    private String version;
    private String description;
    private String url;
    private String checksum;
    private boolean enabled;
}
