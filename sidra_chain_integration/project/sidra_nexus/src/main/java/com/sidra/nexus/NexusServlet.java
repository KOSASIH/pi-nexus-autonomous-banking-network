package com.sidra.nexus;

import org.springframework.boot.web.servlet.ServletRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class NexusServlet {
    @Bean
    public ServletRegistrationBean<NexusServletRegistration> nexusServletRegistration() {
        ServletRegistrationBean<NexusServletRegistration> registration = new ServletRegistrationBean<>(new NexusServletRegistration(), "/nexus/*");
        registration.setName("NexusServlet");
        return registration;
    }
}
