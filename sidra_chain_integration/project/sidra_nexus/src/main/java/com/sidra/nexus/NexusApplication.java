package com.sidra.nexus;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.embedded.jetty.JettyServletWebServerFactory;
import org.springframework.boot.web.servlet.server.ServletWebServerFactory;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class NexusApplication {
    public static void main(String[] args) {
        SpringApplication.run(NexusApplication.class, args);
    }

    @Bean
    public ServletWebServerFactory servletWebServerFactory() {
        JettyServletWebServerFactory factory = new JettyServletWebServerFactory();
        factory.setPort(8081);
        return factory;
    }
}
