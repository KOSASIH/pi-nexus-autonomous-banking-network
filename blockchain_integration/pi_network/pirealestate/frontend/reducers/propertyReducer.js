import { handleActions } from "redux-actions";
import { propertyTypes } from "../types";
import { initialState } from "./initialState";

const {
  GET_PROPERTY_LIST_SUCCESS,
  GET_PROPERTY_DETAILS_SUCCESS,
  CREATE_PROPERTY_SUCCESS,
  UPDATE_PROPERTY_SUCCESS,
  DELETE_PROPERTY_SUCCESS,
} = propertyTypes;

const propertyReducer = handleActions(
  {
    [GET_PROPERTY_LIST_SUCCESS]: (state, action) => {
      return { ...state, propertyList: action.payload };
    },
    [GET_PROPERTY_DETAILS_SUCCESS]: (state, action) => {
      return { ...state, propertyDetails: action.payload };
    },
    [CREATE_PROPERTY_SUCCESS]: (state, action) => {
      return { ...state, propertyList: [...state.propertyList, action.payload] };
    },
    [UPDATE_PROPERTY_SUCCESS]: (state, action) => {
      const updatedPropertyList = state.propertyList.map((property) => {
        if (property.id === action.payload.id) {
          return action.payload;
        }
        return property;
      });
      return { ...state, propertyList: updatedPropertyList };
    },
    [DELETE_PROPERTY_SUCCESS]: (state, action) => {
      return {
        ...state,
        propertyList: state.propertyList.filter((property) => property.id !== action.payload),
      };
    },
  },
  initialState
);

export default propertyReducer;
