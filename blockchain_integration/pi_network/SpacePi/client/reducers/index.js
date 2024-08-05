const initialState = {
  launches: [],
  merchandise: [],
  users: [],
};

export default (state = initialState, action) => {
  switch (action.type) {
    case 'FETCH_LAUNCHES_SUCCESS':
      return {...state, launches: action.launches };
    case 'FETCH_MERCHANDISE_SUCCESS':
      return {...state, merchandise: action.merchandise };
    case 'FETCH_USERS_SUCCESS':
      return {...state, users: action.users };
    default:
      return state;
  }
};
