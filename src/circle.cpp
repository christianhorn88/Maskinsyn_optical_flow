#include "circle.h"

Circle::Circle()
    : Circle(Eigen::Vector2f::Zero(), 0.0)
{ }

Circle::Circle(const Eigen::Vector2f& center, float radius)
    : center_{center}
    , radius_{radius}
{ }

Circle::Circle(const Eigen::Vector2f& p1,
               const Eigen::Vector2f& p2,
               const Eigen::Vector2f& p3)
{
  // p1 and p2 define line_1 as their center line
  Eigen::Vector2f m1 = 0.5f*(p1 + p2);
  Eigen::Vector2f q1{m1.x() + p2.y() - p1.y(), m1.y() - p2.x() + p1.x()};
  Eigen::Vector3f line_1 = m1.homogeneous().cross(q1.homogeneous());

  // p2 and p3 define the line_2 as their center line
  Eigen::Vector2f m2 = 0.5f*(p2 + p3);
  Eigen::Vector2f q2{m2.x() + p3.y() - p2.y(), m2.y() - p3.x() + p2.x()};
  Eigen::Vector3f line_2 = m2.homogeneous().cross(q2.homogeneous());

  // Determine circle
  center_ = line_1.cross(line_2).hnormalized();
  radius_ = (p1 - center_).norm();
}

const Eigen::Vector2f& Circle::center() const
{
  return center_;
}

Eigen::VectorXf Circle::distance(const Eigen::Matrix2Xf& point) const
{
  return (((point.colwise() - center_).colwise().norm()).array() - radius_).abs();
}

double Circle::radius() const
{
  return radius_;
}
