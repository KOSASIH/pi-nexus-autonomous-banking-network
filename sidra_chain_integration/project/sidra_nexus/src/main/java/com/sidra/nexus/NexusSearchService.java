package com.sidra.nexus;

import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class NexusSearchService {
    private final RestHighLevelClient elasticSearchClient;

    @Autowired
    public NexusSearchService(RestHighLevelClient elasticSearchClient) {
        this.elasticSearchClient = elasticSearchClient;
    }

    public List<NexusArtifact> searchArtifacts(String query) {
        SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
        searchSourceBuilder.query(QueryBuilders.matchQuery("name", query));

        SearchRequest searchRequest = new SearchRequest("nexus-artifacts");
        searchRequest.source(searchSourceBuilder);

        try {
            SearchResponse searchResponse = elasticSearchClient.search(searchRequest, RequestOptions.DEFAULT);
            return extractArtifactsFromResponse(searchResponse);
        } catch (IOException e) {
            throw new RuntimeException("Error searching for artifacts", e);
        }
    }

    private List<NexusArtifact> extractArtifactsFromResponse(SearchResponse searchResponse) {
        List<NexusArtifact> artifacts = new ArrayList<>();
        for (SearchHit hit : searchResponse.getHits().getHits()) {
            NexusArtifact artifact = new NexusArtifact();
            artifact.setName(hit.getSourceAsMap().get("name").toString());
            artifact.setGroupId(hit.getSourceAsMap().get("groupId").toString());
            artifact.setArtifactId(hit.getSourceAsMap().get("artifactId").toString());
            artifacts.add(artifact);
        }
        return artifacts;
    }
              }
