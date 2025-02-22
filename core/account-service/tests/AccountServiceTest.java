// AccountServiceTest.java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.boot.test.context.SpringBootTest;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@SpringBootTest
@Testcontainers
@ExtendWith(MockitoExtension.class)
class AccountServiceTest {
    @Container
    private static final PostgreSQLContainer postgreSQLContainer = new PostgreSQLContainer("postgres:11.1");

    @Mock
    private AccountRepository accountRepository;

    @InjectMocks
    private AccountService accountService;

    @BeforeEach
    public void setup() {
        // Initialize test data
    }

    @Test
    void testGetAccount() {
        // Given
        Long accountId = 1L;
        Account account = new Account(accountId, "John Doe", "john.doe@example.com");

        // When
        when(accountRepository.getAccount(accountId)).thenReturn(account);
        Account result = accountService.getAccount(accountId);

        // Then
        assertEquals(account, result);
    }

    @Test
    void testCreateAccount() {
        // Given
        Account account = new Account("John Doe", "john.doe@example.com");

        // When
        accountService.createAccount(account);

        // Then
        verify(accountRepository, times(1)).createAccount(account);
    }
}
