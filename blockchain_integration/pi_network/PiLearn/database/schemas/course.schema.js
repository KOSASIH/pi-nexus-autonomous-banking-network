const Joi = require('joi');

const courseSchema = Joi.object().keys({
  title: Joi.string().required(),
  description: Joi.string().required(),
  price: Joi.number().required(),
  creator: Joi.objectId().required()
});

module.exports = courseSchema;
