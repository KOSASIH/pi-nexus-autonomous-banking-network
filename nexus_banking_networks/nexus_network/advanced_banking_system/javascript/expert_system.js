// expert_system.js
import * as expertSystem from 'expert-system';

const knowledgeBase = [
  {
    condition: 'temperature > 30',
    action: 'turn on air conditioner',
  },
  {
    condition: 'temperature < 20',
    action: 'turn on heater',
  },
];

const expertSystemInstance = new expertSystem(knowledgeBase);

const userInput = {
  temperature: 25,
};

const recommendation = expertSystemInstance.reason(userInput);
console.log(recommendation);
