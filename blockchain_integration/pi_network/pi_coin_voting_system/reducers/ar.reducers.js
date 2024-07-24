const initialState = {
  data: [],
  model: null,
  loading: false,
  error: null,
};

export default (state = initialState, action) => {
  switch (action.type) {
    case 'AR_DATA_LOADING':
      return { ...state, loading: true };
    case 'AR_DATA_RECEIVED':
      return { ...state, data: action.data, loading: false };
    case 'AR_DATA_ERROR':
      return { ...state, error: action.error, loading: false };
    case 'AR_MODEL_LOADING':
      return { ...state, loading: true };
    case 'AR_MODEL_RECEIVED':
      return { ...state, model: action.model, loading: false };
    case 'AR_MODEL_ERROR':
      return { ...state, error: action.error, loading: false };
    case 'AR_MODEL_UPLOADING':
      return { ...state, loading: true };
    case 'AR_MODEL_UPLOADED':
      return { ...state, model: action.model, loading: false };
    case 'AR_MODEL_UPLOAD_ERROR':
      return { ...state, error: action.error, loading: false };
    default:
      return state;
  }
};
