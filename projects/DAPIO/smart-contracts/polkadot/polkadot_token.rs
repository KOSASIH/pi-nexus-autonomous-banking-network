use substrate::{decl_module, decl_storage, decl_event, ensure, StorageValue, StorageMap};
use substrate::traits::{Get, OnInitialize};
use substrate::system::ensure_signed;

pub trait Trait: system::Trait {
    type Event: From<Event<Self>> + Into<<Self as system::Trait>::Event>;
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
    fn deposit_event() {
            <Module<T>>::deposit_event(T::Event::TokenCreated(token_id));
        }

        fn create_token(origin, token_id: u32, token_uri: Vec<u8>) -> DispatchResult {
            ensure_signed(origin)?;
            let token_owner = T::Lookup::lookup(origin)?;
            <Tokens<T>>::insert(token_id, Token { owner: token_owner, token_uri });
            Self::deposit_event();
            Ok(())
        }

        fn transfer_token(origin, token_id: u32, to: T::Lookup) -> DispatchResult {
            ensure_signed(origin)?;
            let token_owner = T::Lookup::lookup(origin)?;
            let token = <Tokens<T>>::get(token_id).ok_or("Token not found")?;
            ensure!(token.owner == token_owner, "Only the owner can transfer");
            <Tokens<T>>::insert(token_id, Token { owner: to, token_uri: token.token_uri });
            Self::deposit_event();
            Ok(())
        }
    }
}

decl_storage! {
    trait Store for Module<T: Trait> as PolkadotToken {
        Tokens get(tokens): map u32 => Token<T>;
    }
}

decl_event! {
    pub enum Event<T> where AccountId = <T as system::Trait>::AccountId {
        TokenCreated(u32),
        TokenTransferred(u32, AccountId),
    }
}

pub struct Token<T: Trait> {
    pub owner: T::Lookup,
    pub token_uri: Vec<u8>,
}
