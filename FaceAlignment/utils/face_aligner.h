#ifndef FEATPIPE_FACE_ALIGNER
#define FEATPIPE_FACE_ALIGNER

#include <string>
#include <map>
#include <opencv2/opencv.hpp>

#include "../FaceAlignment.h"

using cv::Mat;
using cv::Rect;
using cv::Size;
//using cv::Point2d;
using cv::Point2f;
using std::map;

using std::string;

class FaceAligner {
public:
  FaceAligner(const string& model) {
    regressor_.Load(model);
  }

  virtual Mat align(const Mat im, const Rect detection);

  map<string, Point2f> last_landmarks;
  map<string, Point2f> last_gt_landmarks;

protected:
  BoundingBox rectToBb_(Rect rect);
  ShapeRegressor regressor_;

  Size dest_sz_ = Size(125,160);
  Point2f offset_pct_ = Point2f(0.42, 0.45);

};

#endif
