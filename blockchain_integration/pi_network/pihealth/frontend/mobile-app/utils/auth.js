import { FlowClient } from '@onflow/fcl-js';

const flowClient = new FlowClient();

export const authenticate = async () => {
  try {
    const user = await flowClient.authenticate();
    return user;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const getAccessToken = async () => {
  try {
    const token = await flowClient.getAccessToken();
    return token;
  } catch (error) {
    console.error(error);
    return null;
  }
};
