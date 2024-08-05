import { createSelector } from "reselect";
import { getPropertyList } from "../selectors/propertySelectors";
import { getMarketData } from "../selectors/marketSelectors";
import { calculatePropertyValue } from "../utils/valuationAlgorithms";

const getPropertyValuation = createSelector(
  [getPropertyList, getMarketData],
  (propertyList, marketData) => {
    const valuationResults = propertyList.map((property) => {
      const { location, type, size } = property;
      const marketTrends = marketData[location];
      const valuation = calculatePropertyValue(type, size, marketTrends);
      return { ...property, valuation };
    });
    return valuationResults;
  }
);

const getAveragePropertyValuation = createSelector(
  [getPropertyValuation],
  (valuationResults) => {
    const totalValuation = valuationResults.reduce((acc, curr) => acc + curr.valuation, 0);
    return totalValuation / valuationResults.length;
  }
);

const getPropertyValuationDistribution = createSelector(
  [getPropertyValuation],
  (valuationResults) => {
    const distribution = {};
    valuationResults.forEach((result) => {
      const valuationRange = getValuationRange(result.valuation);
      distribution[valuationRange] = (distribution[valuationRange] || 0) + 1;
    });
    return distribution;
  }
);

const getValuationRange = (valuation) => {
  if (valuation < 100000) return "Under $100,000";
  if (valuation < 500000) return "$100,000 - $500,000";
  if (valuation < 1000000) return "$500,000 - $1,000,000";
  return "Over $1,000,000";
};

export { getPropertyValuation, getAveragePropertyValuation, getPropertyValuationDistribution };
