import api from './api';

export const authService = {
  /**
   * Register a new user
   * @param {Object} userData - { username, email, password }
   */
  signup: async (userData) => {
    return await api.post('/auth/signup', userData);
  },

  /**
   * Login user
   * @param {Object} credentials - { email, password }
   */
  login: async (credentials) => {
    return await api.post('/auth/login', credentials);
  },

  /**
   * Get current user
   */
  getCurrentUser: async () => {
    return await api.get('/auth/me');
  }
};
