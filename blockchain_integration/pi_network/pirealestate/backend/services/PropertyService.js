const Property = require("../models/Property");
const UserService = require("./UserService");
const { v4: uuidv4 } = require("uuid");
const AWS = require("aws-sdk");
const s3 = new AWS.S3({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION,
});

class PropertyService {
  async createProperty(data) {
    const property = new Property(data);
    try {
      await property.save();
      return property;
    } catch (error) {
      throw error;
    }
  }

  async getProperty(id) {
    try {
      const property = await Property.findById(id);
      if (!property) {
        throw new Error("Property not found");
      }
      return property;
    } catch (error) {
      throw error;
    }
  }

  async updateProperty(id, data) {
    try {
      const property = await Property.findByIdAndUpdate(id, data, { new: true });
      if (!property) {
        throw new Error("Property not found");
      }
      return property;
    } catch (error) {
      throw error;
    }
  }

  async deleteProperty(id) {
    try {
      await Property.findByIdAndRemove(id);
    } catch (error) {
      throw error;
    }
  }

  async uploadImage(file) {
    const uuid = uuidv4();
    const params = {
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: `properties/${uuid}/${file.originalname}`,
      Body: file.buffer,
    };
    try {
      await s3.upload(params).promise();
      return `https://${process.env.AWS_BUCKET_NAME}.s3.amazonaws.com/properties/${uuid}/${file.originalname}`;
    } catch (error) {
      throw error;
    }
  }

  async getPropertyByOwner(ownerId) {
    try {
      const properties = await Property.find({ owner: ownerId });
      return properties;
    } catch (error) {
      throw error;
    }
  }

  async getPropertyByLocation(location) {
    try {
      const properties = await Property.find({ location });
      return properties;
    } catch (error) {
      throw error;
    }
  }
}

module.exports = new PropertyService();
