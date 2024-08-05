// food_safety.js
import { trackEvent } from './tracker';
import { FOOD_SAFETY_EVENTS } from '../constants/analyticsEvents';

const foodSafetyAnalytics = {
  trackFoodSafetyIssue: (issueType, foodId) => {
    trackEvent(FOOD_SAFETY_EVENTS.FOOD_SAFETY_ISSUE, {
      issueType,
      foodId,
    });
  },

  trackFoodRecall: (foodId, reason) => {
    trackEvent(FOOD_SAFETY_EVENTS.FOOD_RECALL, {
      foodId,
      reason,
    });
  },

  trackFoodContamination: (foodId, contaminant) => {
    trackEvent(FOOD_SAFETY_EVENTS.FOOD_CONTAMINATION, {
      foodId,
      contaminant,
    });
  },

  trackFoodborneIllness: (foodId, illnessType) => {
    trackEvent(FOOD_SAFETY_EVENTS.FOOD_BORNE_ILLNESS, {
      foodId,
      illnessType,
    });
  },
};

export default foodSafetyAnalytics;
