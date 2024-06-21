// UserService.java
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    private final RedisTemplate<String, User> redisTemplate;
    private final UserRepository userRepository;
    private final JwtAuthenticationToken jwtAuthenticationToken;

    public UserService(RedisTemplate<String, User> redisTemplate, UserRepository userRepository, JwtAuthenticationToken jwtAuthenticationToken) {
        this.redisTemplate = redisTemplate;
        this.userRepository = userRepository;
        this.jwtAuthenticationToken = jwtAuthenticationToken;
    }

    public User findByUsername(String username) {
        String cacheKey = "user:username:" + username;
        User cachedUser = redisTemplate.opsForValue().get(cacheKey);
        if (cachedUser!= null) {
            return cachedUser;
        }
        User user = userRepository.findByUsername(username);
        redisTemplate.opsForValue().set(cacheKey, user);
        return user;
    }

    public User findByEmail(String email) {
        String cacheKey = "user:email:" + email;
        User cachedUser = redisTemplate.opsForValue().get(cacheKey);
        if (cachedUser!= null) {
            return cachedUser;
        }
        User user = userRepository.findByEmail(email);
        redisTemplate.opsForValue().set(cacheKey, user);
        return user;
    }
}
