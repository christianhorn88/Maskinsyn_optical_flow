// Make shorthand alias for logical vector.
using LogicalVector = Eigen::Matrix<bool, 1, Eigen::Dynamic>;

/// \brief Datatype for ransac estimate as a result from RansacEstimator.
struct RansacEstimate
{
  Eigen::RowVector2f best_vector;
  int num_iterations;
  Eigen::Index num_inliers;
    cv::Mat is_inler_mat;
};

/// \brief A robust vector estimator based on optical flow vectors.
class RansacEstimator
{
public:
  /// \brief Constructs a vector estimator.
  /// \param p The desired probability of getting a good sample.
  /// \param distance_threshold The maximum distance a good sample can have from the mean vector.
  explicit RansacEstimator(double p = 0.99, float distance_threshold = 0.5f, float angle_threshold = 0.5f, float min_vector_length_threshold = 2.0f, int pixel_jump = 10);

    /// \brief Estimates a vector based on the vector measurements using RANSAC.
    /// \param gradient vector measurements of optical flow corrupted by noise.
    /// \return The vector estimate based on the entire inlier set.
    RansacEstimate estimate(const cv::Mat pts) const;

    /// \brief Estimates a vector based on the vector measurements using RANSAC on masked pixels
    /// \param gradient vector measurements of optical flow corrupted by noise.
    /// \return The vector estimate based on the entire inlier set.
    RansacEstimate estimate_with_mask(const cv::Mat pts, const cv::Mat mask) const;

private:

    double p_;
    float distance_threshold_;
    float angle_threshold_;
    float min_vector_length_threshold_;
    int pixel_jump_;  //Only evaluate every X'th pixel for speed optimization
};
