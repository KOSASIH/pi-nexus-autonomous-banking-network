import express from 'express';
import api from './api';
import wallet from './wallet';
import dex from './dex';

const app = express();

app.use('/api', api);
app.use('/wallet', wallet);
app.use('/dex', dex);

app.listen(3000, () => {
    console.log('Pi DApp listening on port 3000');
});
