package com.sidra.nexus;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface NexusArtifactRepository extends JpaRepository<NexusArtifact, Long> {
    List<NexusArtifact> findByGroupIdAndArtifactId(String groupId, String artifactId);
    List<NexusArtifact> findByEnabledTrue();
}
