import { createContext, useState, useEffect } from 'react';
import UserService from '../services/UserService';

const UserContext = createContext();

const UserProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const authenticate = async () => {
            try {
                const user = await UserService.authenticate();
                setUser(user);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        authenticate();
    }, []);

    const createUser = async (username, password) => {
        try {
            const user = await UserService.createUser(username, password);
            setUser(user);
        } catch (err) {
            setError(err.message);
        }
    };

    const deleteUser = async () => {
        try {
            await UserService.deleteUser();
            setUser(null);
        } catch (err) {
            setError(err.message);
        }
    };

    return (
        <UserContext.Provider value={{ user, loading, error, createUser, deleteUser }}>
            {children}
        </UserContext.Provider>
    );
};

export { UserProvider, UserContext };
