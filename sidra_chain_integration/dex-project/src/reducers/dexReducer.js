import { handleActions } from "redux-actions";

const initialState = {
    orders: [],
};

const dexReducer = handleActions(
    {
        PLACE_ORDER: (state, action) => {
            // Handle place order action
        },
        CANCEL_ORDER: (state, action) => {
            // Handle cancel order action
        },
    },
    initialState
);

export default dexReducer;
