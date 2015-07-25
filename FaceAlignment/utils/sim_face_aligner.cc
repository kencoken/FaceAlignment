#include "sim_face_aligner.h"

#include <iostream>

#include "../FaceAlignment.h"

Mat SimFaceAligner::align(const Mat im, Rect detection) {
  BoundingBox bb = rectToBb_(detection);

  const int initial_number = 20;
  Mat_<double> current_shape = regressor_.Predict(im, bb, initial_number);

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
  std::cout << M << std::endl;
  cv::warpAffine(im, transformed_im, M, Size(500, 500));

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
