use substrate::dispatch::{DispatchError, DispatchResult};
use substrate::storage::{StorageMap, StorageValue};
use substrate::traits::{OnFinalize, OnInitialize};
use substrate::{decl_event, decl_module, decl_storage, ensure};

pub trait Trait: substrate::Trait {
    type Event: From<Event<Self>> + Into<<Self as substrate::Trait>::Event>;
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
        fn deposit(origin, amount: u128) -> DispatchResult {
            ensure!(origin == T::Origin::signed(T::AccountId::from([1; 32])), "Only the relay chain can deposit");
            let who = T::AccountId::from([1; 32]);
            <balances::Module<T>>::inc_balance(&who, amount);
            Ok(())
        }

        fn withdraw(origin, amount: u128) -> DispatchResult {
            ensure!(origin == T::Origin::signed(T::AccountId::from([1; 32])), "Only the relay chain can withdraw");
            let who = T::AccountId::from([1; 32]);
            <balances::Module<T>>::dec_balance(&who, amount);
            Ok(())
        }
    }
}

decl_storage! {
    trait Store for Module<T: Trait> as PolkadotStorage {
        get balance: map [T::AccountId; u128];
    }
}

decl_event! {
pub enum Event<T> where AccountId = <T as substrate::Trait>::AccountId {
    Deposit(AccountId, u128),
    Withdrawal(AccountId, u128),
}
  }
