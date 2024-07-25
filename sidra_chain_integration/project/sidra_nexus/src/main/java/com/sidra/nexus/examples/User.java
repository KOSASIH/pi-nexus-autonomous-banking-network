package com.sidra.nexus.model;

import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.util.List;

@Entity
@Table(name = "users")
public class User {
    @Id
    private Long id;
    private String username;
    private String password;
    private List<String> roles;

    // getters and setters
}
