// src/dashboard/dashboard.ts
import { Server } from 'socket.io';

const io = new Server(3000);

io.on('connection', (socket) => {
    console.log('New client connected');
    socket.on('disconnect', () => {
        console.log('Client disconnected');
    });
});

export const emitTransactionData = (data: any) => {
    io.emit('transactionUpdate', data);
};
