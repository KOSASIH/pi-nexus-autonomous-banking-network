package com.sidra.nexus;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;

public class NexusApplication {
    public static void main(String[] args) throws Exception {
        Server server = new Server(8081);
        ServletContextHandler context = new ServletContextHandler(server, "/", ServletContextHandler.SESSIONS);
        context.addServlet(new ServletHolder(new NexusServlet()), "/*");
        server.start();
        server.join();
    }
}
