import api from './api';

export const feedbackService = {
  submitFeedback: (feedbackData) => {
    // feedbackData should match FeedbackRequest schema:
    // { prediction, prediction_confidence, verification_status, evidence_score, similarity_score, final_decision, user_feedback, comment }
    return api.post('/feedback', feedbackData);
  },

  getFeedbacks: () => {
    return api.get('/feedback');
  }
};

export default feedbackService;
