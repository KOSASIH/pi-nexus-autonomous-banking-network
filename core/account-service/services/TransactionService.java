// TransactionService.java
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.messaging.Sink;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import io.opentracing.Tracer;

@Service
@EnableBinding(Sink.class)
public class TransactionService {
    private final TransactionRepository transactionRepository;
    private final Sink sink;
    private final Tracer tracer;

    public TransactionService(TransactionRepository transactionRepository, Sink sink, Tracer tracer) {
        this.transactionRepository = transactionRepository;
        this.sink = sink;
        this.tracer = tracer;
    }

    @Transactional
    @StreamListener
    public void createTransaction(Transaction transaction) {
        Span span = tracer.buildSpan("createTransaction").start();
        try {
            transactionRepository.createTransaction(transaction);
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
            transactionRepository.updateTransaction(transaction);
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
