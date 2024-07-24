import * as Twitter from 'twitter';
import * as Facebook from 'facebook-js-sdk';
import * as Instagram from 'instagram-js-sdk';

class AstralPlaneSocial {
  constructor() {
    this.twitter = new Twitter({
      consumer_key: 'consumerkey',
      consumer_secret: 'consumersecret',
      access_token_key: 'accesstokenkey',
      access_token_secret: 'accesstokensecret',
    });
    this.facebook = new Facebook({
      appId: 'appid',
      appSecret: 'appsecret',
    });
    this.instagram = new Instagram({
      clientId: 'clientid',
      clientSecret: 'clientsecret',
    });
  }

  async tweet(message) {
    await this.twitter.post('statuses/update', { status: message });
  }

  async getTweets() {
    const tweets = await this.twitter.get('statuses/home_timeline');
    return tweets;
  }

  async shareOnFacebook(message) {
    await this.facebook.api('/me/feed', 'post', { message });
  }

  async getFacebookPosts() {
    const posts = await this.facebook.api('/me/posts');
    return posts;
  }

  async shareOnInstagram(image, caption) {
    await this.instagram.media.upload(image, { caption });
  }

  async getInstagramPosts() {
    const posts = await this.instagram.media.recent();
    return posts;
  }

  async shareAssetOnSocialMedia(asset) {
    const message = `Check out this new asset on Astral Plane: ${asset.name}!`;
    await this.tweet(message);
    await this.shareOnFacebook(message);
    await this.shareOnInstagram(asset.image, message);
  }
}

export default AstralPlaneSocial;
