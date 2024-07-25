package com.sidra.nexus;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class NexusArtifactController {
    private final NexusArtifactService artifactService;

    @Autowired
    public NexusArtifactController(NexusArtifactService artifactService) {
        this.artifactService = artifactService;
    }

    @GetMapping("/artifacts")
    public ResponseEntity<List<NexusArtifact>> getArtifacts() {
        return new ResponseEntity<>(artifactService.getArtifacts(), HttpStatus.OK);
    }

    @GetMapping("/artifacts/{id}")
    public ResponseEntity<NexusArtifact> getArtifact(@PathVariable Long id) {
        return new ResponseEntity<>(artifactService.getArtifact(id), HttpStatus.OK);
    }

    @PostMapping("/artifacts")
    public ResponseEntity<NexusArtifact> createArtifact(@RequestBody NexusArtifact artifact) {
        return new ResponseEntity<>(artifactService.createArtifact(artifact), HttpStatus.CREATED);
    }

    @PutMapping("/artifacts/{id}")
    public ResponseEntity<NexusArtifact> updateArtifact(@PathVariable Long id, @RequestBody NexusArtifact artifact) {
        return new ResponseEntity<>(artifactService.updateArtifact(artifact), HttpStatus.OK);
    }

    @DeleteMapping("/artifacts/{id}")
    public ResponseEntity<Void> deleteArtifact(@PathVariable Long id) {
        artifactService.deleteArtifact(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }
}
