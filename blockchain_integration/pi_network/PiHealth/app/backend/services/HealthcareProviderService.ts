import { HealthcareProviderModel } from '../models/HealthcareProvider';

class HealthcareProviderService {
  async createHealthcareProvider(healthcareProvider: HealthcareProvider) {
    try {
      const newHealthcareProvider = new HealthcareProviderModel(healthcareProvider);
      await newHealthcareProvider.save();
      return newHealthcareProvider;
    } catch (error) {
      throw error;
    }
  }

  async getHealthcareProvider(id: string) {
    try {
      const healthcareProvider = await HealthcareProviderModel.findById(id);
      return healthcareProvider;
    } catch (error) {
      throw error;
    }
  }

  async updateHealthcareProvider(id: string, healthcareProvider: HealthcareProvider) {
    try {
      const updatedHealthcareProvider = await HealthcareProviderModel.findByIdAndUpdate(id, healthcareProvider, { new: true });
      return updatedHealthcareProvider;
    } catch (error) {
      throw error;
    }
  }

  async deleteHealthcareProvider(id: string) {
    try {
      await HealthcareProviderModel.findByIdAndRemove(id);
    } catch (error) {
      throw error;
    }
  }

  async getHealthcareProviders() {
    try {
      const healthcareProviders = await HealthcareProviderModel.find();
      return healthcareProviders;
    } catch (error) {
      throw error;
    }
  }
}

export default HealthcareProviderService;
