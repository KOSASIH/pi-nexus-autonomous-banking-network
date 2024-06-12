// reducers/browserReducer.js
import { combineReducers } from 'redux';
import { COMPILE_CODE, SAVE_CODE } from '../actions/browserActions';

const initialState = {
  compiledCode: '',
  htmlCode: '',
  cssCode: '',
  jsCode: '',
};

const browserReducer = (state = initialState, action) => {
  switch (action.type) {
    case COMPILE_CODE:
      return { ...state, compiledCode: action.compiledCode };
    case SAVE_CODE:
      return { ...state, htmlCode: action.htmlCode, cssCode: action.cssCode, jsCode: action.jsCode };
    default:
      return state;
  }
};

export default browserReducer;
