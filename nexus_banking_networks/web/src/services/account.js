import api from "./api";

export async function getAccounts() {
  const response = await api.get("/accounts");
  return response.data;
}
