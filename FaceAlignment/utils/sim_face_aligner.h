#ifndef FEATPIPE_SIM_FACE_ALIGNER
#define FEATPIPE_SIM_FACE_ALIGNER

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "../FaceAlignment.h"

using cv::Mat;
using cv::Rect;
using cv::Size;
//using cv::Point2d;
using cv::Point2f;
using cv::Mat_;

using std::string;
using std::vector;
using std::map;
using std::ifstream;

class SimFaceAligner {
public:
  SimFaceAligner(const string& model,
                 const string& shape_samples_gt,
                 const string& bounding_box_gt) {
    regressor_.Load(model);

    mean_shape_ = getMeanShape_(shape_samples_gt, bounding_box_gt);

  }

  virtual Mat align(const Mat im, const Rect detection);

  vector<Point2f> last_landmarks;
  vector<Point2f> last_gt_landmarks;

protected:
  Mat_<double> getMeanShape_(const string& shape_samples_gt,
                             const string& bounding_box_gt);
  BoundingBox rectToBb_(Rect rect);

  ShapeRegressor regressor_;
  Mat_<double> mean_shape_;

  Size dest_sz_ = Size(125,160);

  float scale_ = 0.75;
  float top_pct = 0.5;

};

#endif
