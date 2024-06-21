// AccountDAO.java
import com.zaxxer.hikari.HikariDataSource;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Repository;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Repository
public class AccountDAO {
    private final HikariDataSource dataSource;
    private final Logger logger;

    public AccountDAO(HikariDataSource dataSource) {
        this.dataSource = dataSource;
        this.logger = LogManager.getLogger(AccountDAO.class);
    }

    @Retryable(value = Exception.class, maxAttempts = 3, backoff = @Backoff(delay = 500))
    public Account getAccount(Long accountId) {
        // JDBC code to retrieve account
        logger.info("Retrieving account with ID {}", accountId);
    }
}
