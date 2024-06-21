// UserRepository.java
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends Neo4jRepository<User, Long> {
    @Query("MATCH (u:User {username: $username}) RETURN u")
    User findByUsername(String username);

    @Query("MATCH (u:User {email: $email}) RETURN u")
    User findByEmail(String email);
}

// UserRepositoryImpl.java
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.stereotype.Repository;

@Repository
public class UserRepositoryImpl implements UserRepository {
    private final RedisTemplate<String, User> redisTemplate;
    private final UserRepository userRepository;
    private final JwtAuthenticationToken jwtAuthenticationToken;

    public UserRepositoryImpl(RedisTemplate<String, User> redisTemplate, UserRepository userRepository, JwtAuthenticationToken jwtAuthenticationToken) {
        this.redisTemplate = redisTemplate;
        this.userRepository = userRepository;
        this.jwtAuthenticationToken = jwtAuthenticationToken;
    }

    public User findByUsername(String username) {
        String cacheKey = "user:username:" + username;
        User cachedUser = redisTemplate.opsForValue().get(cacheKey);
        if (cachedUser != null) {
            return cachedUser;
        }
        User user = userRepository.findByUsername(username);
        redisTemplate.opsForValue().set(cacheKey, user);
        return user;
    }

    public User findByEmail(String email) {
        String cacheKey = "user:email:" + email;
        User cachedUser = redisTemplate.opsForValue().get(cacheKey);
        if (cachedUser != null) {
            return cachedUser;
        }
        User user = userRepository.findByEmail(email);
        redisTemplate.opsForValue().set(cacheKey, user);
        return user;
    }
            }
