import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.time.Time;

public class FraudDetector {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Read streaming data from Kafka topic
        DataStream<String> transactions = env.addSource(new FlinkKafkaConsumer<>("transactions", new SimpleStringSchema(), props));

        // Map transactions to a tuple of (user_id, amount)
        DataStream<Tuple2<String, Double>> transactionTuples = transactions.map(new MapFunction<String, Tuple2<String, Double>>() {
            @Override
            public Tuple2<String, Double> map(String transaction) throws Exception {
                String[] fields = transaction.split(",");
                return new Tuple2<>(fields[0], Double.parseDouble(fields[1]));
            }
        });

        // Windowing and aggregation to detect fraud
        DataStream<Tuple2<String, Double>> fraudScores = transactionTuples
               .keyBy(0) // user_id
               .timeWindow(Time.seconds(30)) // 30-second window
               .reduce(new ReduceFunction<Tuple2<String, Double>>() {
                    @Override
                    public Tuple2<String, Double> reduce(Tuple2<String, Double> t1, Tuple2<String, Double> t2) throws Exception {
                        // Calculate fraud score based on transaction amount and frequency
                        double fraudScore = t1.f1 + t2.f1;
                        return new Tuple2<>(t1.f0, fraudScore);
                    }
                });

        // Alert on high fraud scores
        fraudScores.filter(new FilterFunction<Tuple2<String, Double>>() {
            @Override
            public boolean filter(Tuple2<String, Double> t) throws Exception {
                return t.f1 > 100; // threshold for high fraud score
            }
        }).print();

        env.execute("Fraud Detector");
    }
}
