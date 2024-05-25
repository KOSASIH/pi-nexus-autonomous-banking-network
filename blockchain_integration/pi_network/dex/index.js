import DEX from './DEX';
import DEXController from './DEXController';

const dex = new DEX();
const dexController = new DEXController(dex);

export default dexController;
