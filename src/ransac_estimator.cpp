#include "ransac_estimator.h"
#include <random>




RansacEstimator::RansacEstimator(double p, float distance_threshold, float angle_threshold, float min_vector_length_threshold, int pixel_jump)
  : p_{p}
  , distance_threshold_{distance_threshold}
  , angle_threshold_{angle_threshold}
  , min_vector_length_threshold_{min_vector_length_threshold}
  , pixel_jump_{pixel_jump}   //Only evaluate every X'th pixel for speed optimization

{}

RansacEstimate RansacEstimator::estimate(const cv::Mat pts) const
{
  // Initialize best set.
  Eigen::Index best_num_inliers{0};
  Eigen::RowVector2f best_vector;
  cv::Mat best_inliers_mat = cv::Mat::zeros(pts.rows/pixel_jump_,pts.cols/pixel_jump_, CV_8UC1);

  // Set up random number generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> uni_dist_x(0, static_cast<int>((pts.cols-1)/pixel_jump_));
  std::uniform_int_distribution<> uni_dist_y(0, static_cast<int>((pts.rows-1)/pixel_jump_));

  // Initialize maximum number of iterations.
  int max_iterations = std::numeric_limits<int>::max();
  int iterations{0};

// Check that flow vector has 10 valid vectors (vectors with minimum length of min_vector_length_threshold_)
  bool flow_ok = false;
  int valid_points = 0;
  for (int y = 0; y <= (pts.rows-1); y += pixel_jump_) {
    for (int x = 0; x <= (pts.cols - 1); x += pixel_jump_) {
      volatile const cv::Point2f flowatxy = pts.at<cv::Point2f>(y, x);
      if ((std::abs(flowatxy.x) + std::abs(flowatxy.y)) >= min_vector_length_threshold_) {
        valid_points++;
        if (valid_points >= 10) {
          flow_ok = true;
          y = pts.rows;
          x = pts.cols;
        }
      }
    }
  }

  if (flow_ok) {

    // Perform RANSAC.
    for (; iterations < max_iterations; ++iterations) {
      // Determine test vector.
      // Loop until a vector with minimum length is found
      bool not_found = true;
      int x_gen;
      int y_gen;
      while (not_found) {
        y_gen = uni_dist_y(gen);
        x_gen = uni_dist_x(gen);
        volatile const cv::Point2f flowatxy_test = pts.at<cv::Point2f>(y_gen * pixel_jump_, x_gen * pixel_jump_);
        if (std::abs(flowatxy_test.x) + std::abs(flowatxy_test.y) >= min_vector_length_threshold_) {
          not_found = false;
        }
      }
      volatile const cv::Point2f flowatxy_test = pts.at<cv::Point2f>(y_gen * pixel_jump_, x_gen * pixel_jump_);

      // Count number of inliers.
      cv::Mat is_inlier_mat = cv::Mat::zeros(pts.rows / pixel_jump_, pts.cols / pixel_jump_, CV_8UC1);
      int counter = 0;
      double test_num_inliers = 0;

      for (int y = 0; y <= (pts.rows - 1); y += pixel_jump_) {
        for (int x = 0; x <= (pts.cols - 1); x += pixel_jump_) {
          volatile const cv::Point2f flowatxy = pts.at<cv::Point2f>(y, x);
          if ((std::abs(std::atan2(flowatxy_test.x, flowatxy_test.y) - std::atan2(flowatxy.x, flowatxy.y)) <= angle_threshold_) &
              (std::abs(flowatxy_test.x - flowatxy.x) <= distance_threshold_) &
              (std::abs(flowatxy_test.y - flowatxy.y) <= distance_threshold_)) {
            is_inlier_mat.col(x / pixel_jump_).row(y / pixel_jump_) = 255;
            test_num_inliers++;
          }
          counter++;
        }
      }

      // Check if this estimate gave a better result.
      if (test_num_inliers > best_num_inliers) {
        // Update vector with largest inlier set.
        best_vector[0] = flowatxy_test.x;
        best_vector[1] = flowatxy_test.y;
        best_num_inliers = test_num_inliers;
        best_inliers_mat = is_inlier_mat;

        // Update max iterations.
        double inlier_ratio =
          static_cast<double>(best_num_inliers) / static_cast<double>(pts.cols * pts.rows / pixel_jump_ / pixel_jump_);
        max_iterations = static_cast<int>(std::log(1.0 - p_) /
                                          std::log(1.0 - inlier_ratio * inlier_ratio * inlier_ratio));
        if (max_iterations > 10000) {max_iterations = 1000;} //Limit iterations for speed optimization
      }
    }
    //cout << "best_num_inliers: " << best_num_inliers << "  best vector: " << best_vector << " \t iterations: " << iterations << "  best_inliers: " << "\t \n";

  }
  else {cout << "RANSAC 1: Too little flow! \n";}

  return {best_vector, iterations, best_num_inliers, best_inliers_mat};
}






RansacEstimate RansacEstimator::estimate_with_mask(const cv::Mat pts, const cv::Mat mask) const
{
  // Initialize best set.
  Eigen::Index best_num_inliers{0};
  Eigen::RowVector2f best_vector;
  cv::Mat best_inliers_mat = cv::Mat::zeros(pts.rows/pixel_jump_,pts.cols/pixel_jump_, CV_8UC1);

  // Set up random number generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> uni_dist_x(0, static_cast<int>((pts.cols-1)/pixel_jump_));
  std::uniform_int_distribution<> uni_dist_y(0, static_cast<int>((pts.rows-1)/pixel_jump_));

  // Initialize maximum number of iterations.
  int max_iterations = std::numeric_limits<int>::max();
  int iterations{0};

  // Check that mask vector has 10 valid points
  bool mask_ok = false;
  int valid_points = 0;
  for (int y = 0; y <= (mask.rows-1); y++) {
    for (int x = 0; x <= (mask.cols - 1); x++) {
      if (mask.at<char>(y, x) == 0) {
        valid_points++;
        if (valid_points >= 10) {
          mask_ok = true;
          y = mask.rows;
          x = mask.cols;
        }
      }
    }
  }

  // Check that masked flow vector has 10 valid vectors (vectors with minimum length of min_vector_length_threshold_)
  bool flow_ok = false;
  valid_points = 0;
  for (int y = 0; y <= (pts.rows-1); y += pixel_jump_) {
    for (int x = 0; x <= (pts.cols - 1); x += pixel_jump_) {
      if (mask.at<char>(y/pixel_jump_, x/pixel_jump_) == 0) {
        volatile const cv::Point2f flowatxy = pts.at<cv::Point2f>(y, x);
        if ((std::abs(flowatxy.x) + std::abs(flowatxy.y)) >= min_vector_length_threshold_) {
          valid_points++;
          if (valid_points >= 10) {
            flow_ok = true;
            y = pts.rows;
            x = pts.cols;
          }
        }
      }
    }
  }

  if (mask_ok & flow_ok) {
    // Perform RANSAC.
    for (; iterations < max_iterations; ++iterations) {
      // Determine test vector.
      // Loop until a masked vector with minimum length is found
      bool not_found = true;
      int x_gen;
      int y_gen;
      while (not_found) {
        y_gen = uni_dist_y(gen);
        x_gen = uni_dist_x(gen);
        volatile const cv::Point2f flowatxy_test = pts.at<cv::Point2f>(y_gen * pixel_jump_, x_gen * pixel_jump_);
        if ((mask.at<char>(y_gen, x_gen) == 0) &
           (std::abs(flowatxy_test.x) + std::abs(flowatxy_test.y) >= min_vector_length_threshold_)) {
          not_found = false;
        }
      }
      volatile const cv::Point2f flowatxy_test = pts.at<cv::Point2f>(y_gen * pixel_jump_, x_gen * pixel_jump_);


      // Count number of inliers.
      cv::Mat is_inlier_mat = cv::Mat::zeros(pts.rows / pixel_jump_, pts.cols / pixel_jump_, CV_8UC1);
      int counter = 0;
      double test_num_inliers = 0;

      for (int y = 0; y <= (pts.rows - 1) / pixel_jump_; y += 1) {
        for (int x = 0; x <= (pts.cols - 1) / pixel_jump_; x += 1) {
          if (mask.at<char>(y, x) == 0) {
            volatile const cv::Point2f flowatxy = pts.at<cv::Point2f>(y * 10, x * 10);
            if ((std::abs(std::atan2(flowatxy_test.x,flowatxy_test.y) - std::atan2(flowatxy.x,flowatxy.y)) <= angle_threshold_) &
                (std::abs(flowatxy_test.x - flowatxy.x) <= distance_threshold_) &
                (std::abs(flowatxy_test.y - flowatxy.y) <= distance_threshold_)) {
              is_inlier_mat.col(x).row(y) = 255;
              test_num_inliers++;
            }
            counter++;
          }
        }
      }

      // Check if this estimate gave a better result.
      if (test_num_inliers > best_num_inliers) {
        // Update vector with largest inlier set.
        best_vector[0] = flowatxy_test.x;
        best_vector[1] = flowatxy_test.y;
        best_num_inliers = test_num_inliers;
        best_inliers_mat = is_inlier_mat;

        // Update max iterations.
        double inlier_ratio = static_cast<double>(best_num_inliers) / static_cast<double>(counter);
        max_iterations = static_cast<int>(std::log(1.0 - p_) /
                                          std::log(1.0 - inlier_ratio * inlier_ratio * inlier_ratio));
        if (max_iterations > 10000) {max_iterations = 1000;} //Limit iterations for speed optimization
      }
    }
    cout << "best_num_inliers: " << best_num_inliers << "  best vector: " << best_vector << " \t iterations: "
         << iterations << "  best_inliers: " << "\t \n";
    //cv::bitwise_not (best_inliers_mat, best_inliers_mat );
  }
  else if (!mask_ok) {cout << "RANSAC 2: Empty mask! \n";}
  else {cout << "RANSAC 2: Too little flow! \n";}

  return {best_vector, iterations, best_num_inliers, best_inliers_mat};
}