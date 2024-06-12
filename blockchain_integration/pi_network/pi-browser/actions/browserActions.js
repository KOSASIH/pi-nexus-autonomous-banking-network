// actions/browserActions.js
import { createAction } from 'redux-actions';
import { codeCompilerAPI } from '../api/codeCompilerAPI';

export const COMPILE_CODE = 'COMPILE_CODE';
export const SAVE_CODE = 'SAVE_CODE';

export const compileCode = createAction(COMPILE_CODE, (htmlCode, cssCode, jsCode) => ({ htmlCode, cssCode, jsCode }));
export const saveCode = createAction(SAVE_CODE, (htmlCode, cssCode, jsCode) => ({ htmlCode, cssCode, jsCode }));

export const compileAndSaveCode = (htmlCode, cssCode, jsCode) => async (dispatch) => {
  try {
    const response = await codeCompilerAPI.compileCode(htmlCode, cssCode, jsCode);
    dispatch(compileCode(response.data.compiledCode));
    dispatch(saveCode(htmlCode, cssCode, jsCode));
  } catch (error) {
    console.error(error);
  }
};
