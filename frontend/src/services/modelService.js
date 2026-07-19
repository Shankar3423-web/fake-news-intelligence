import api from './api';

export const modelService = {
  getModelInfo: () => {
    return api.get('/model/info');
  },

  getHealth: () => {
    return api.get('/health');
  },

  triggerRetraining: () => {
    return api.post('/retrain');
  }
};

export default modelService;
