import * as uport from 'uport-credentials';

const useIdentityVerification = () => {
  const [ identity, setIdentity ] = useState(null);

  useEffect(() => {
    const verifyIdentity = async () => {
      const credentials = await uport.getCredentials();
      const identity = await uport.verifyCredentials(credentials);
      setIdentity(identity);
    };
    verifyIdentity();
  }, []);

  return { verify };
};

export default useIdentityVerification;
