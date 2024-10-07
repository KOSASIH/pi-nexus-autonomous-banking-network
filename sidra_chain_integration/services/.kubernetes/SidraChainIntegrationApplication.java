import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
@RestController
public class SidraChainIntegrationApplication {

  @GetMapping("/sidra-chain-integration")
  public String getSidraChainIntegration() {
    return "Sidra Chain Integration Service";
  }

  public static void main(String[] args) {
    SpringApplication.run(SidraChainIntegrationApplication.class, args);
  }
}
