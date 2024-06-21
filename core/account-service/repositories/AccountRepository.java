// AccountRepository.java
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Repository;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;

@Repository
public class AccountRepository {
    private final RedisTemplate<String, Account> redisTemplate;
    private final AccountDAO accountDAO;

    public AccountRepository(RedisTemplate<String, Account> redisTemplate, AccountDAO accountDAO) {
        this.redisTemplate = redisTemplate;
        this.accountDAO = accountDAO;
    }

    @CircuitBreaker(name = "accountService", fallbackMethod = "fallbackGetAccount")
    public Account getAccount(Long accountId) {
        String cacheKey = "account:" + accountId;
        Account cachedAccount = redisTemplate.opsForValue().get(cacheKey);
        if (cachedAccount!= null) {
            return cachedAccount;
        }
        Account account = accountDAO.getAccount(accountId);
        redisTemplate.opsForValue().set(cacheKey, account);
        return account;
    }

    private Account fallbackGetAccount(Long accountId, Throwable throwable) {
        // Return a default account or throw an exception
        return new Account();
    }
}
