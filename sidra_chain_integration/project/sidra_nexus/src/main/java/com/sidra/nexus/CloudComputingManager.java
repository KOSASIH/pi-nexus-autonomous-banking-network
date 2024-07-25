package com.sidra.nexus;

import org.jclouds.ContextBuilder;
import org.jclouds.openstackSwift.v1.SwiftApi;

public class CloudComputingManager {
    private SwiftApi swiftApi;

    public CloudComputingManager() {
        swiftApi = ContextBuilder.newBuilder("swift")
                .credentials("username", "password")
                .buildApi(SwiftApi.class);
    }

    public void uploadFile(String container, String file) {
        swiftApi.putObject(container, file);
    }

    public void downloadFile(String container, String file) {
        swiftApi.getObject(container, file);
    }

    public void createContainer(String container) {
        swiftApi.createContainer(container);
    }
}
