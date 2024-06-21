// AccountDAO.java
import com.zaxxer.hikari.HikariDataSource;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Repository;

@Repository
public class AccountDAO {
    private final HikariDataSource dataSource;

    public AccountDAO(HikariDataSource dataSource) {
        this.dataSource = dataSource;
    }

    @Retryable(value = Exception.class, maxAttempts = 3, backoff = @Backoff(delay = 500))
    public Account getAccount(Long accountId) {
        // JDBC code to retrieve account
    }
}
