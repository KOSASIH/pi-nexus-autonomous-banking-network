import api from './api'

export async function getTransactions() {
  const response = await api.get('/transactions')
  return response.data
}
