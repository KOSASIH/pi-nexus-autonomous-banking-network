package com.sidra.nexus;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.PerspectiveCamera;
import javafx.scene.Scene;
import javafx.scene.shape.Sphere;
import javafx.stage.Stage;

public class VirtualRealityManager extends Application {
    @Override
    public void start(Stage primaryStage) {
        Group group = new Group();
        Scene scene = new Scene(group, 600, 600, true);
        PerspectiveCamera camera = new PerspectiveCamera(true);
        scene.setCamera(camera);

        Sphere sphere = new Sphere(100);
        group.getChildren().add(sphere);

        primaryStage.setTitle("Virtual Reality");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
