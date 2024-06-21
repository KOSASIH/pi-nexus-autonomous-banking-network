// AccountService.java
import org.springframework.cloud.client.circuitbreaker.CircuitBreaker;
import org.springframework.cloud.client.circuitbreaker.CircuitBreakerFactory;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Service
@EnableDiscoveryClient
@EnableEurekaClient
public class AccountService {
    private final RedisTemplate<String, Account> redisTemplate;
    private final CircuitBreakerFactory circuitBreakerFactory;
    private final AccountRepository accountRepository;
    private final DiscoveryClient discoveryClient;

    public AccountService(RedisTemplate<String, Account> redisTemplate, CircuitBreakerFactory circuitBreakerFactory, AccountRepository accountRepository, DiscoveryClient discoveryClient) {
        this.redisTemplate = redisTemplate;
        this.circuitBreakerFactory = circuitBreakerFactory;
        this.accountRepository = accountRepository;
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
            Account account = accountRepository.getAccount(accountId);
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
