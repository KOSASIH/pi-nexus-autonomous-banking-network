export async function fetchAPI(endpoint: string, method: string, data: any): Promise<any> {
  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${getLocalStorageItem(STORAGE_KEYS.AUTH_TOKEN)}`,
  };

  const response = await fetch(`${API_URL}/${endpoint}`, {
    method,
    headers,
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(response.statusText);
  }

  return response.json();
}
