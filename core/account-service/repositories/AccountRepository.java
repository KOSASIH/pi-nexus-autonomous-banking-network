// AccountRepository.java
import org.springframework.cloud.client.circuitbreaker.CircuitBreaker;
import org.springframework.cloud.client.circuitbreaker.CircuitBreakerFactory;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Repository;

@Repository
@EnableDiscoveryClient
@EnableEurekaClient
public class AccountRepository {
    private final RedisTemplate<String, Account> redisTemplate;
    private final CircuitBreakerFactory circuitBreakerFactory;
    private final AccountDAO accountDAO;
    private final DiscoveryClient discoveryClient;

    public AccountRepository(RedisTemplate<String, Account> redisTemplate, CircuitBreakerFactory circuitBreakerFactory, AccountDAO accountDAO, DiscoveryClient discoveryClient) {
        this.redisTemplate = redisTemplate;
        this.circuitBreakerFactory = circuitBreakerFactory;
        this.accountDAO = accountDAO;
        this.discoveryClient = discoveryClient;
    }

    public Account getAccount(Long accountId) {
        CircuitBreaker circuitBreaker = circuitBreakerFactory.create("accountService");
        return circuitBreaker.run(() -> {
            String cacheKey = "account:" + accountId;
            Account cachedAccount = redisTemplate.opsForValue().get(cacheKey);
            if (cachedAccount != null) {
                return cachedAccount;
            }
            Account account = accountDAO.getAccount(accountId);
            redisTemplate.opsForValue().set(cacheKey, account);
            return account;
        }, throwable -> {
            // Fallback logic
            return new Account();
        });
    }

    @HystrixCommand(fallbackMethod = "fallbackGetAccount")
    public Account getAccountFallback(Long accountId) {
        // Fallback logic
        return new Account();
    }
    }
