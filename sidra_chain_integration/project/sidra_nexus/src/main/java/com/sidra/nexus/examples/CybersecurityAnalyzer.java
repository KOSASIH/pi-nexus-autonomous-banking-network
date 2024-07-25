public class Main {
    public static void main(String[] args) {
        CybersecurityAnalyzer analyzer = new CybersecurityAnalyzer();

        try {
            analyzer.analyzeInput("Hello, World!");
        } catch (ValidationException | EncodingException | IntrusionException e) {
            System.out.println("Error: " + e.getMessage());
        }

        analyzer.scanForMalware("example.exe");
        analyzer.monitorNetworkTraffic();
    }
}
