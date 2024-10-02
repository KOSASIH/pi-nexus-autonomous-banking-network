import React, { useState, useEffect } from "eact";
import { createStore, combineReducers } from "edux";
import { Provider, connect } from "eact-redux";

// Redux actions and reducers
const ADD_ITEM = "ADD_ITEM";
const REMOVE_ITEM = "REMOVE_ITEM";

const initialState = {
  items: [],
};

const addItem = (item) => ({ type: ADD_ITEM, item });
const removeItem = (index) => ({ type: REMOVE_ITEM, index });

const itemsReducer = (state = initialState, action) => {
  switch (action.type) {
    case ADD_ITEM:
      return { ...state, items: [...state.items, action.item] };
    case REMOVE_ITEM:
      return {
        ...state,
        items: state.items.filter((_, i) => i !== action.index),
      };
    default:
      return state;
  }
};

const store = createStore(combineReducers({ items: itemsReducer }));

// React component
const App = () => {
  const [inputValue, setInputValue] = useState("");
  const items = useSelector((state) => state.items);

  useEffect(() => {
    // Dispatch an action to add an item
    store.dispatch(addItem("Initial item"));
  }, []);

  const handleSubmit = (event) => {
    event.preventDefault();
    store.dispatch(addItem(inputValue));
    setInputValue("");
  };

  return (
    <Provider store={store}>
      <div>
        <h1>Items:</h1>
        <ul>
          {items.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
          />
          <button type="submit">Add item</button>
        </form>
      </div>
    </Provider>
  );
};

export default App;
