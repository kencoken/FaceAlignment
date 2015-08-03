#include "sim_face_aligner.h"

#include <iostream>

#include "../FaceAlignment.h"

Mat SimFaceAligner::align(const Mat im, Rect detection) {
  BoundingBox bb = rectToBb_(detection);

  const int initial_number = 20;
  Mat_<double> current_shape = regressor_.Predict(im, bb, initial_number);
  last_landmarks.clear();
  for (size_t i = 0; i < current_shape.rows; ++i) {
    last_landmarks.push_back(Point2f(current_shape.at<double>(i, 0),
                                     current_shape.at<double>(i, 1)));
  }

  // start alignment

  Mat_<double> rotation;
  double scale;

  Mat_<double> projected_shape = ProjectShape(current_shape, rectToBb_(detection));
  std::cout << "mean shape is: " << mean_shape_.rows << "x" << mean_shape_.cols << std::endl;
  std::cout << "normalized shape is: " << projected_shape.rows << "x" << projected_shape.cols << std::endl;
  SimilarityTransform(mean_shape_, projected_shape,
                      rotation, scale);

  std::cout << "rotation is: " << rotation << std::endl;
  std::cout << "scale is: " << scale << std::endl;

  Mat transformed_im;
  //transpose(rotation, rotation);
  Mat_<double> M = Mat::zeros(2, 3, CV_64F);
  rotation.copyTo(M(Rect(0, 0, 2, 2)));
  M.at<double>(0,0) = M.at<double>(0,0)*scale;
  M.at<double>(1,1) = M.at<double>(1,1)*scale;

  int top_margin = 100;
  int side_margin = 50;
  int bottom_margin = 50;

  vector<Point2f> intermediary_landmarks;
  cv::transform(last_landmarks, intermediary_landmarks, M);

  float mean_x = 0, mean_y = 0;
  for (size_t i = 0; i < intermediary_landmarks.size(); ++i) {
    mean_x += intermediary_landmarks[i].x;
    mean_y += intermediary_landmarks[i].y;
  }
  mean_x /= static_cast<float>(intermediary_landmarks.size());
  mean_y /= static_cast<float>(intermediary_landmarks.size());
  int top_offset = static_cast<int>(top_pct*static_cast<float>(dest_sz_.height) - mean_y);
  int left_offset = static_cast<int>(0.5*static_cast<float>(dest_sz_.width) - mean_x);

  M.at<double>(0,2) = M.at<double>(0,2) + left_offset;
  M.at<double>(1,2) = M.at<double>(1,2) + top_offset;

  Mat M_rotsc = cv::getRotationMatrix2D(Point2f(dest_sz_.width/2.0, dest_sz_.height/2.0),
                                        0.0, scale_);

  /* compose transform */

  Mat identity_row = Mat::zeros(1, 3, CV_64F);
  identity_row.at<double>(0,2) = 1.0;

  M.push_back(identity_row);
  M_rotsc.push_back(identity_row);
  M = M_rotsc*M;
  M = M(Rect(0, 0, 3, 2));

  std::cout << M << std::endl;
  cv::warpAffine(im, transformed_im, M, Size(500, 500));

  transformed_im = transformed_im(cv::Rect(0, 0,
                                           dest_sz_.width,
                                           dest_sz_.height));
  std::cout << "Size is: " << transformed_im.cols << "x" << transformed_im.rows << std::endl;

  cv::transform(last_landmarks, last_gt_landmarks, M);

  return transformed_im;

}

Mat_<double> SimFaceAligner::getMeanShape_(const string& shape_samples_gt,
                                           const string& bounding_box_gt) {

  // load shapes

  ifstream shapef(shape_samples_gt);
  string line;
  vector<Mat_<double> > shapes;
  while(std::getline(shapef, line)) {
    vector<string> strs;
    boost::split(strs, line, boost::is_any_of("\t"));

    Mat_<double> shape = Mat(strs.size(), 2, CV_64F);
    for (auto i = 0; i < strs.size(); ++i) {
      vector<string> coords;
      boost::split(coords, strs[i], boost::is_any_of(","));
      assert(coords.size() == 2);
      shape.at<double>(i, 0) = boost::lexical_cast<double>(coords[0]);
      shape.at<double>(i, 1) = boost::lexical_cast<double>(coords[1]);
    }
    shapes.push_back(shape);
  }

  // load bounding boxes

  ifstream bbf(bounding_box_gt);
  vector<BoundingBox> bbs;
  while(std::getline(bbf, line)) {
    vector<string> strs;
    boost::split(strs, line, boost::is_any_of("\t"));
    assert(strs.size() == 4);

    Rect bb_rect(boost::lexical_cast<int>(strs[0]),
                 boost::lexical_cast<int>(strs[1]),
                 boost::lexical_cast<int>(strs[2]),
                 boost::lexical_cast<int>(strs[3]));
    bbs.push_back(rectToBb_(bb_rect));
  }

  std::cout << bbs.size() << "," << shapes.size() << std::endl;
  assert(bbs.size() == shapes.size());

  // compute mean

  std::cout << shapes.size() << std::endl;
  std::cout << shapes[0].rows << "x" << shapes[0].cols << std::endl;
  return GetMeanShape(shapes, bbs);

}

BoundingBox SimFaceAligner::rectToBb_(Rect rect) {
  BoundingBox bb;
  bb.start_x = rect.x;
  bb.start_y = rect.y;
  bb.width = rect.width;
  bb.height = rect.height;
  bb.centroid_x = rect.x + rect.width / 2.0;
  bb.centroid_y = rect.y + rect.height / 2.0;
  return bb;
}
