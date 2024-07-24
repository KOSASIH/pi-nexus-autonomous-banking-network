import * as Phaser from 'phaser';

class AstralPlaneGame {
  constructor() {
    this.game = new Phaser.Game({
      type: Phaser.CANVAS,
      parent: 'game-container',
      width: 800,
      height: 600,
      scene: {
        preload: this.preload,
        create: this.create,
        update: this.update,
      },
    });
  }

  preload() {
    this.game.load.image('asset-image', 'https://example.com/image.jpg');
  }

  create() {
    const assetImage = this.game.add.image(400, 300, 'asset-image');
    assetImage.setScale(2);
  }

  update(time, delta) {
    // Update game logic here
  }
}

export default AstralPlaneGame;
