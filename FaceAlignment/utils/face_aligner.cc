#include "face_aligner.h"

#include <vector>
#include <cmath>

using cv::Mat_;

using std::vector;

Mat FaceAligner::align(const Mat im, Rect detection) {

  BoundingBox bb = rectToBb_(detection);

  const int initial_number = 20;
  Mat_<double> current_shape = regressor_.Predict(im, bb, initial_number);

  // get eyes and nosetip

  map<string, Point2f> landmarks;

  int i = 0;
  i = 10; landmarks["eye_left"] = Point2f(current_shape(i,0), current_shape(i,1));
  i = 11; landmarks["eye_right"] = Point2f(current_shape(i,0), current_shape(i,1));
  i = 21; landmarks["nose_tip"] = Point2f(current_shape(i,0), current_shape(i,1));

  // get cardinal points

  map<string, Point2f> landmarks_gt;

  landmarks_gt["eye_left"] = Point2f(dest_sz_.width*offset_pct_.x,
                                     dest_sz_.height*offset_pct_.y);
  landmarks_gt["eye_right"] = Point2f(dest_sz_.width*(1.0 - offset_pct_.x),
                                      dest_sz_.height*offset_pct_.y);

  // compute rigid transform

  Mat M = Mat::eye(2, 3, CV_64F);

  /* translate */
  M.at<double>(0, 2) = landmarks_gt["eye_left"].x - landmarks["eye_left"].x;
  M.at<double>(1, 2) = landmarks_gt["eye_left"].y - landmarks["eye_left"].y;

  /* rotate + scale */
  double x_offset = landmarks["eye_right"].x - landmarks["eye_left"].x;
  double y_offset = landmarks["eye_right"].y - landmarks["eye_left"].y;
  double hyp = sqrt(x_offset*x_offset + y_offset*y_offset);
  double ang = std::acos(x_offset/hyp)/M_PI*180.0;
  if (y_offset < 0) ang *= -1.0;
  std::cout << "Angle is: " << ang << std::endl;

  double x_offset_gt = landmarks_gt["eye_right"].x - landmarks_gt["eye_left"].x;
  double y_offset_gt = landmarks_gt["eye_right"].y - landmarks_gt["eye_left"].y;
  double hyp_gt = sqrt(x_offset_gt*x_offset_gt + y_offset_gt*y_offset_gt);

  double scale = hyp_gt/hyp;

  Mat M_rotsc = cv::getRotationMatrix2D(landmarks_gt["eye_left"], ang, scale);

  /* compose transform */

  Mat identity_row = Mat::zeros(1, 3, CV_64F);
  identity_row.at<double>(0,2) = 1.0;

  M.push_back(identity_row);
  M_rotsc.push_back(identity_row);
  M = M_rotsc*M;
  M = M(Rect(0, 0, 3, 2));

  cv::Mat transformed_im;
  cv::warpAffine(im, transformed_im, M, dest_sz_);

  last_landmarks = landmarks;
  last_gt_landmarks = landmarks_gt;

  return transformed_im;

}

BoundingBox FaceAligner::rectToBb_(Rect rect) {
  BoundingBox bb;
  bb.start_x = rect.x;
  bb.start_y = rect.y;
  bb.width = rect.width;
  bb.height = rect.height;
  bb.centroid_x = rect.x + rect.width / 2.0;
  bb.centroid_y = rect.y + rect.height / 2.0;
  return bb;
}
