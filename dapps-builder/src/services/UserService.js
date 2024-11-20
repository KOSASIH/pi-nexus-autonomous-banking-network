class UserService {
    constructor() {
        this.users = new Map(); // In-memory user storage for demonstration
    }

    createUser(username, password) {
        if (this.users.has(username)) {
            throw new Error('User already exists.');
        }
        this.users.set(username, { password });
        return { username };
    }

    authenticate(username, password) {
        const user = this.users.get(username);
        if (!user || user.password !== password) {
            throw new Error('Invalid username or password.');
        }
        return { username };
    }

    getUser(username) {
        if (!this.users.has(username)) {
            throw new Error('User not found.');
        }
        return { username };
    }

    deleteUser(username) {
        if (!this.users.has(username)) {
            throw new Error('User not found.');
        }
        this.users.delete(username);
        return { username };
    }
}

export default new UserService();
