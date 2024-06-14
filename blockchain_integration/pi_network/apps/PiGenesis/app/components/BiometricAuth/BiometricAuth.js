import axios from 'axios';

const BiometricAuth = () => {
  const [biometricData, setBiometricData] = useState(null);
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    // Initialize biometric authentication module
    const biometricAuthModule = new BiometricAuthModule();

    biometricAuthModule.init()
     .then(() => {
        // Get biometric data from user
        biometricAuthModule.getBiometricData()
         .then((data) => {
            setBiometricData(data);
          })
         .catch((error) => {
            console.error(error);
          });
      })
     .catch((error) => {
        console.error(error);
      });
  }, []);

  const handleAuthenticate = () => {
    axios.post('/api/authenticate', biometricData)
     .then((response) => {
        setAuthenticated(true);
      })
     .catch((error) => {
        console.error(error);
      });
  };

  return (
    <div>
      {authenticated? (
        <p>Authenticated!</p>
      ) : (
        <div>
          <h2>Biometric Authentication</h2>
          <p>Place your finger on the sensor or look into the camera to authenticate.</p>
          <button onClick={handleAuthenticate}>Authenticate</button>
        </div>
      )}
    </div>
  );
};

export default BiometricAuth;
