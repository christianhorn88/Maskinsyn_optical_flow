#ifndef LAB_3_CORNERS_CIRCLE_H
#define LAB_3_CORNERS_CIRCLE_H

#include "Eigen/Eigen"

struct Circle
{
  /// \brief Constructs an empty circle.
  Circle();

  /// \brief Constructs a circle from a center point and a radius.
  /// \param center The center point of the circle.
  /// \param radius The radius of the circle.
  Circle(const Eigen::Vector2f& center, float radius);

  /// \brief Constructs a circle from three points on the circle.
  /// \param p1 First point on the circle.
  /// \param p2 Second point on the circle.
  /// \param p3 Third point on the circle.
  Circle(const Eigen::Vector2f& p1,
         const Eigen::Vector2f& p2,
         const Eigen::Vector2f& p3);

  /// \brief The center point of the circle.
  /// \return The center point.
  const Eigen::Vector2f& center() const;

  /// \brief The distance between the circle and a point.
  /// \return The distance.
  Eigen::VectorXf distance(const Eigen::Matrix2Xf& point) const;

  /// \brief The circle radius.
  /// \return The radius.
  double radius() const;

private:
  Eigen::Vector2f center_;
  float radius_;
};

#endif //LAB_3_CORNERS_CIRCLE_H
