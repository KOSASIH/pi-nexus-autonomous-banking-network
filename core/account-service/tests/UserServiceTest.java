// UserServiceTest.java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.boot.test.context.SpringBootTest;
import org.testcontainers.containers.Neo4jContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@SpringBootTest
@Testcontainers
@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    @Container
    private static final Neo4jContainer neo4jContainer = new Neo4jContainer("neo4j:4.2.1");

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    @BeforeEach
    public void setup() {
        // Initialize test data
    }

    @Test
    void testFindByUsername() {
        // Given
        String username = "john.doe";
        User user = new User(username, "John Doe", "john.doe@example.com");

        // When
        when(userRepository.findByUsername(username)).thenReturn(user);
        User result = userService.findByUsername(username);

        // Then
        assertEquals(user, result);
    }

    @Test
    void testFindByEmail() {
        // Given
        String email = "john.doe@example.com";
        User user = new User("john.doe", "John Doe", email);

        // When
        when(userRepository.findByEmail(email)).thenReturn(user);
        User result = userService.findByEmail(email);

        // Then
        assertEquals(user, result);
    }
}
