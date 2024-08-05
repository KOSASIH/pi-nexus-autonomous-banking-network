import axios from "axios";
import { createAction } from "redux-actions";
import { API_URL } from "../constants";
import { propertyTypes } from "../types";

// Create actions
export const getPropertyList = createAction("GET_PROPERTY_LIST");
export const getPropertyDetails = createAction("GET_PROPERTY_DETAILS");
export const createProperty = createAction("CREATE_PROPERTY");
export const updateProperty = createAction("UPDATE_PROPERTY");
export const deleteProperty = createAction("DELETE_PROPERTY");

// Action creators
export function fetchPropertyList() {
  return async (dispatch) => {
    try {
      const response = await axios.get(`${API_URL}/properties`);
      dispatch(getPropertyList(response.data));
    } catch (error) {
      console.error(error);
    }
  };
}

export function fetchPropertyDetails(id) {
  return async (dispatch) => {
    try {
      const response = await axios.get(`${API_URL}/properties/${id}`);
      dispatch(getPropertyDetails(response.data));
    } catch (error) {
      console.error(error);
    }
  };
}

export function createNewProperty(property) {
  return async (dispatch) => {
    try {
      const response = await axios.post(`${API_URL}/properties`, property);
      dispatch(createProperty(response.data));
    } catch (error) {
      console.error(error);
    }
  };
}

export function updateExistingProperty(id, property) {
  return async (dispatch) => {
    try {
      const response = await axios.put(`${API_URL}/properties/${id}`, property);
      dispatch(updateProperty(response.data));
    } catch (error) {
      console.error(error);
    }
  };
}

export function deletePropertyById(id) {
  return async (dispatch) => {
    try {
      await axios.delete(`${API_URL}/properties/${id}`);
      dispatch(deleteProperty(id));
    } catch (error) {
      console.error(error);
    }
  };
}
