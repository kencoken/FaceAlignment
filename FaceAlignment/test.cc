#include "utils/face_aligner.h"
#include "utils/sim_face_aligner.h"

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

int main() {
  // load in images etc. (taken from TestDemo.cpp)
  vector<Mat_<uchar> > test_images;
  vector<BoundingBox> test_bounding_box;
  int test_img_num = 507;
  int initial_number = 20;
  int landmark_num = 29;
  ifstream fin;

  for(int i = 0;i < test_img_num;i++){
    string image_name = "../data/COFW_Dataset/testImages/";
    image_name = image_name + to_string(i+1) + ".jpg";
    Mat_<uchar> temp = imread(image_name,0);
    test_images.push_back(temp);
  }
  fin.open("../data/COFW_Dataset/boundingbox_test.txt");
  for(int i = 0;i < test_img_num;i++){
    BoundingBox temp;
    fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
    temp.centroid_x = temp.start_x + temp.width/2.0;
    temp.centroid_y = temp.start_y + temp.height/2.0;
    test_bounding_box.push_back(temp);
  }
  fin.close();

  // setup face aligner

  FaceAligner fa("../data/model.txt");
  SimFaceAligner fas("../data/model.txt",
                     "../data/COFW_Dataset/shapes.txt",
                     "../data/COFW_Dataset/boundingbox.txt");

  // vis

  while(true){
    int index = 1;
    cout<<"Input index:"<<endl;
    cin>>index;

    cv::Rect rect_bb;
    rect_bb.x = test_bounding_box[index].start_x;
    rect_bb.y = test_bounding_box[index].start_y;
    rect_bb.width = test_bounding_box[index].width;
    rect_bb.height = test_bounding_box[index].height;

    Mat in_im = test_images[index];
    Mat im = fa.align(in_im, rect_bb);

    cvtColor(in_im, in_im, CV_GRAY2RGB);
    circle(in_im, fa.last_landmarks["eye_left"], 3, cv::Scalar(0,255,0),-1,8,0);
    circle(in_im, fa.last_landmarks["eye_right"], 3, cv::Scalar(0,0,255),-1,8,0);
    rectangle(in_im, Point(rect_bb.x, rect_bb.y),
              Point(rect_bb.x + rect_bb.width, rect_bb.y + rect_bb.height),
              cv::Scalar(0,255,255),2,8,0);
    imshow("result", in_im);
    waitKey(0);

    cvtColor(im, im, CV_GRAY2RGB);
    circle(im, fa.last_gt_landmarks["eye_left"], 3, cv::Scalar(0,255,0),-1,8,0);
    circle(im, fa.last_gt_landmarks["eye_right"], 3, cv::Scalar(0,0,255),-1,8,0);
    imshow("result", im);
    waitKey(0);

    // transform image

    Mat im_fas = fas.align(in_im, rect_bb);
    imshow("Similarity Transform Result", im_fas);
    waitKey(0);

  }
  return 0;

}
