// TransactionServiceTest.java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.boot.test.context.SpringBootTest;
import org.testcontainers.containers.KafkaContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@SpringBootTest
@Testcontainers
@ExtendWith(MockitoExtension.class)
class TransactionServiceTest {
    @Container
    private static final KafkaContainer kafkaContainer = new KafkaContainer("confluentinc/cp-kafka:5.5.1");

    @Mock
    private TransactionRepository transactionRepository;

    @InjectMocks
    private TransactionService transactionService;

    @BeforeEach
    public void setup() {
        // Initialize test data
    }

    @Test
    void testCreateTransaction() {
        // Given
        Transaction transaction = new Transaction("John Doe", 100.0);

        // When
        transactionService.createTransaction(transaction);

        // Then
        verify(transactionRepository, times(1)).createTransaction(transaction);
    }

    @Test
    void testUpdateTransaction() {
        // Given
        Transaction transaction = new Transaction("John Doe", 100.0);

        // When
        transactionService.updateTransaction(transaction);

        // Then
        verify(transactionRepository, times(1)).updateTransaction(transaction);
    }
}
