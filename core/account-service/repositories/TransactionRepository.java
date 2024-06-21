// TransactionRepository.java
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

@Repository
public class TransactionRepository {
    private final TransactionDAO transactionDAO;
    private final KafkaTemplate<String, TransactionEvent> kafkaTemplate;

    public TransactionRepository(TransactionDAO transactionDAO, KafkaTemplate<String, TransactionEvent> kafkaTemplate) {
        this.transactionDAO = transactionDAO;
        this.kafkaTemplate = kafkaTemplate;
    }

    @Transactional
    public void createTransaction(Transaction transaction) {
        transactionDAO.createTransaction(transaction);
        TransactionEvent event = new TransactionEvent(transaction);
        kafkaTemplate.send("transactions", event);
    }

    @Transactional
    public void updateTransaction(Transaction transaction) {
        transactionDAO.updateTransaction(transaction);
        TransactionEvent event = new TransactionEvent(transaction);
        kafkaTemplate.send("transactions", event);
    }
}
