// UserRepository.java
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends Neo4jRepository<User, Long> {
    @Query("MATCH (u:User {username: $username}) RETURN u")
    User findByUsername(String username);

    @Query("MATCH (u:User {email: $email}) RETURN u")
    User findByEmail(String email);
}
