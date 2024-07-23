// sidra_data_vault/src/main.rs
use ipfs::{Ipfs, IpfsApi};
use sodium::{Box_, Key};

struct DataVault {
    ipfs: Ipfs,
    key: Key,
}

impl DataVault {
    fn new(ipfs: Ipfs, key: Key) -> Self {
        DataVault { ipfs, key }
    }

    fn store_data(&self, data: Vec<u8>) -> String {
        // Encrypt the data using the key
        let encrypted_data = Box_::seal(data, &self.key);

        // Store the encrypted data in IPFS
        let cid = self.ipfs.add(encrypted_data).unwrap();
        cid.to_string()
    }

    fn retrieve_data(&self, cid: String) -> Vec<u8> {
        // Retrieve the encrypted data from IPFS
        let encrypted_data = self.ipfs.get(cid).unwrap();

        // Decrypt the data using the key
        let decrypted_data = Box_::open(encrypted_data, &self.key).unwrap();
        decrypted_data
    }
}
