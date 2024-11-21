import java.util.concurrent.*;
import java.util.stream.*;

public class AdvancedConcurrency {

  public static void main(String[] args) {
    // Create an ExecutorService with a fixed thread pool
    ExecutorService executor = Executors.newFixedThreadPool(4);

    // Submit Callable tasks to the ExecutorService
    List<Callable<Integer>> tasks =
        List.of(
            () -> {
              return doWork(1);
            },
            () -> {
              return doWork(2);
            },
            () -> {
              return doWork(3);
            },
            () -> {
              return doWork(4);
            });

    try {
      // Invoke all tasks and collect results in a List<Future<Integer>>
      List<Future<Integer>> results = executor.invokeAll(tasks);

      // Process results
      for (Future<Integer> result : results) {
        System.out.println(result.get());
      }
    } catch (InterruptedException | ExecutionException e) {
      e.printStackTrace();
    } finally {
      // Shutdown the ExecutorService
      executor.shutdown();
    }

    // Parallel Stream example
    int sum = IntStream.range(0, 1000000).parallel().reduce(0, Integer::sum);
    System.out.println("Sum: " + sum);
  }

  private static int doWork(int id) {
    // Simulate some work
    try {
      Thread.sleep(1000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return id * 2;
  }
}
