import AstralPlaneSocial from './social/AstralPlaneSocial';
import AstralPlaneAI from './ai/AstralPlaneAI';
import AstralPlaneBlockchain from './blockchain/AstralPlaneBlockchain';
import AstralPlaneDatabase from './database/AstralPlaneDatabase';
import AstralPlaneGame from './game/AstralPlaneGame';

const social = new AstralPlaneSocial();
const ai = new AstralPlaneAI();
const blockchain = new AstralPlaneBlockchain();
const database = new AstralPlaneDatabase();
const game = new AstralPlaneGame();

async function main() {
  await social.shareAssetOnSocialMedia({ name: 'Test Asset', image: 'https://example.com/image.jpg' });
  const generatedAsset = await ai.generateAsset();
  await blockchain.createAsset(generatedAsset);
  await database.createAsset(generatedAsset);
  game.game.start();
}

main();
