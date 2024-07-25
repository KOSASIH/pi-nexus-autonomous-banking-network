package com.sidra.nexus;

import org.springframework.data.jpa.repository.JpaRepository;

public interface NexusRepository extends JpaRepository<NexusArtifact, Long> {
    List<NexusArtifact> findByGroupIdAndArtifactId(String groupId, String artifactId);
}
