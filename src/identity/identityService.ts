// src/identity/identityService.ts
import { DID } from 'did-library';

export const createIdentity = async (userData: any) => {
    const did = new DID();
    const identity = await did.create(userData);
    return identity;
};
