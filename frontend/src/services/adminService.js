import api from './api';

export const adminService = {
  getPendingReviews: () => {
    return api.get('/admin/pending');
  },

  reviewFeedback: (feedbackId, reviewStatus, reviewer = 'Admin Reviewer', notes = '') => {
    return api.post('/admin/review', {
      feedback_id: String(feedbackId),
      review_status: reviewStatus, // 'APPROVED' or 'REJECTED'
      reviewer,
      notes
    });
  }
};

export default adminService;
