import api from './api';

export const predictionService = {
  predictNews: (title, text) => {
    return api.post('/predict', { title, text });
  },

  analyzeFull: (title, text) => {
    return api.post('/analyze-full', { title, text });
  },

  getHistory: () => {
    return api.get('/predict/history');
  },

  getStats: () => {
    return api.get('/predict/stats');
  }
};

export default predictionService;
