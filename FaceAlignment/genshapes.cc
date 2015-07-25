#include "utils/face_aligner.h"

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

int main() {
  // load in images etc. (taken from TestDemo.cpp)
  vector<Mat_<uchar> > images;
  vector<BoundingBox> bounding_box;
  int img_num = 1345;
  int initial_number = 20;
  int landmark_num = 29;
  ifstream fin;

  for(int i = 0;i < img_num;i++){
    string image_name = "../data/COFW_Dataset/trainingImages/";
    image_name = image_name + to_string(i+1) + ".jpg";
    Mat_<uchar> temp = imread(image_name,0);
    images.push_back(temp);
  }
  fin.open("../data/COFW_Dataset/boundingbox.txt");
  for(int i = 0;i < img_num;i++){
    BoundingBox temp;
    fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
    temp.centroid_x = temp.start_x + temp.width/2.0;
    temp.centroid_y = temp.start_y + temp.height/2.0;
    bounding_box.push_back(temp);
  }
  fin.close();

  // setup computation regressor

  ShapeRegressor regressor;
  regressor.Load("../data/model.txt");

  string output_file = "../data/COFW_Dataset/shapes.txt";

  // compute all shapes

  ofstream fout(output_file);

  for (auto i = 0; i < bounding_box.size(); ++i) {
    cout << "Computing for image " << i+1 << " of " << bounding_box.size() << endl;

    cv::Rect rect_bb;
    rect_bb.x = bounding_box[i].start_x;
    rect_bb.y = bounding_box[i].start_y;
    rect_bb.width = bounding_box[i].width;
    rect_bb.height = bounding_box[i].height;

    const int initial_number = 20;
    Mat_<double> current_shape = regressor.Predict(images[i],
                                                   bounding_box[i],
                                                   initial_number);

    cout << current_shape.rows << "x" << current_shape.cols << endl;
    assert(current_shape.rows > 1);
    assert(current_shape.cols == 2);
    for (auto idx = 0; idx < (current_shape.rows-1); ++idx) {
      fout << current_shape.at<double>(idx, 0) << ","
           << current_shape.at<double>(idx, 1) << "\t";
    }
    fout << current_shape.at<double>(current_shape.rows-1, 0) << ","
         << current_shape.at<double>(current_shape.rows-1, 1) << endl;
  }

  return 0;

}
