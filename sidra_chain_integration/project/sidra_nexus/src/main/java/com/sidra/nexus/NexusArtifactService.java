package com.sidra.nexus;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class NexusArtifactService {
    private final NexusArtifactRepository artifactRepository;

    @Autowired
    public NexusArtifactService(NexusArtifactRepository artifactRepository) {
        this.artifactRepository = artifactRepository;
    }

    public List<NexusArtifact> getArtifacts() {
        return artifactRepository.findAll();
    }

    public NexusArtifact getArtifact(Long id) {
        return artifactRepository.findById(id).orElseThrow();
    }

    public NexusArtifact createArtifact(NexusArtifact artifact) {
        return artifactRepository.save(artifact);
    }

    public NexusArtifact updateArtifact(NexusArtifact artifact) {
        return artifactRepository.save(artifact);
    }

    public void deleteArtifact(Long id) {
        artifactRepository.deleteById(id);
    }
}
