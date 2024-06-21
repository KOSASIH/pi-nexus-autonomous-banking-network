import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Properties;

public class TransactionProcessor {
    private KafkaProducer<String, String> producer;

    public TransactionProcessor(String bootstrapServers) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        producer = new KafkaProducer<>(props);
    }

    public void processTransaction(Transaction transaction) {
        String transactionJson = transaction.toJson();
        ProducerRecord<String, String> record = new ProducerRecord<>("transactions", transactionJson);
        producer.send(record);
    }

    public static void main(String[] args) {
        TransactionProcessor processor = new TransactionProcessor("localhost:9092");
        Transaction transaction = new Transaction("1234567890", 1000, "USA", "VISA");
        processor.processTransaction(transaction);
    }
}

class Transaction {
    private String cardNumber;
    private int amount;
    private String country;
    private String cardType;

    // getters and setters
}
