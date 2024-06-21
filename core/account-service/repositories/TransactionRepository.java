// TransactionRepository.java
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.messaging.Sink;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;
import io.opentracing.Tracer;

@Repository
@EnableBinding(Sink.class)
public class TransactionRepository {
    private final TransactionDAO transactionDAO;
    private final Sink sink;
    private final Tracer tracer;

    public TransactionRepository(TransactionDAO transactionDAO, Sink sink, Tracer tracer) {
        this.transactionDAO = transactionDAO;
        this.sink = sink;
        this.tracer = tracer;
    }

    @Transactional
    @StreamListener
    public void createTransaction(Transaction transaction) {
        Span span = tracer.buildSpan("createTransaction").start();
        try {
            transactionDAO.createTransaction(transaction);
            sink.input().send(MessageBuilder.withPayload(new TransactionEvent(transaction)).build());
            span.setTag("success", true);
        } catch (Exception e) {
            span.setTag("success", false);
            span.log(e.getMessage());
        } finally {
            span.finish();
        }
    }

    @Transactional
    @StreamListener
    public void updateTransaction(Transaction transaction) {
        Span span = tracer.buildSpan("updateTransaction").start();
        try {
            transactionDAO.updateTransaction(transaction);
            sink.input().send(MessageBuilder.withPayload(new TransactionEvent(transaction)).build());
            span.setTag("success", true);
        } catch (Exception e) {
            span.setTag("success", false);
            span.log(e.getMessage());
        } finally {
            span.finish();
        }
    }
            }
