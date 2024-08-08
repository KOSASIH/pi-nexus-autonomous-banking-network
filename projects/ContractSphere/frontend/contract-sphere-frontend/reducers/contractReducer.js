const initialState = {
  contracts: [],
  contract: null,
  error: null
};

export default function contractReducer(state = initialState, action) {
  switch (action.type) {
    case 'FETCH_CONTRACTS_SUCCESS':
      return { ...state, contracts: action.contracts };
    case 'FETCH_CONTRACTS_FAILURE':
      return { ...state, error: action.error };
    case 'FETCH_CONTRACT_SUCCESS':
      return { ...state, contract: action.contract };
    case 'FETCH_CONTRACT_FAILURE':
      return { ...state, error: action.error };
    case 'CREATE_CONTRACT_SUCCESS':
      return { ...state, contract: action.contract };
    case 'CREATE_CONTRACT_FAILURE':
      return { ...state, error: action.error };
    default:
      return state;
  }
}
