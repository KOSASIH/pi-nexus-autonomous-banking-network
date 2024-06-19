class DynamicPricingController {
  async calculatePrice(req, res) {
    const { serviceType, demand, marketCondition } = req.body;
    const price = calculateDynamicPrice(serviceType, demand, marketCondition);
    res.json({ price });
  }
}
