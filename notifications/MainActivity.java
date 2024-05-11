import com.dieam.reactnativepushnotification.modules.RNPushNotification;

public class MainActivity extends AppCompatActivity {
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    // ... other init code ...

    RNPushNotification.IntentHandlers.add(
        new RNPushNotification.RNIntentHandler() {
          @Override
          public void onNewIntent(Intent intent) {
            // If your provider requires some parsing on the intent before the data can be
            // used, add that code here. Otherwise leave empty.
          }

          @Nullable
          @Override
          public Bundle getBundleFromIntent(Intent intent) {
            // This should return the bundle data that will be serialized to the `notification.data`
            // property sent to the `onNotification()` handler. Return `null` if there is no data
            // or this is not an intent from your provider.

            // Example:
            if (intent.hasExtra("MY_NOTIFICATION_PROVIDER_DATA_KEY")) {
              return intent.getBundleExtra("MY_NOTIFICATION_PROVIDER_DATA_KEY");
            }
            return null;
          }
        });
  }
}
