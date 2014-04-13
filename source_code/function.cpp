/** 
 * @author Bi Sai 
 * @version 2014/03/29
 */

#include "face.h"

void train(const vector<Mat_<uchar> >& input_images,                  
        const vector<Mat_<double> >& target_shapes,
        const Mat_<double>& mean_shape,
        int initial_number,
        int pixel_pair_num,
        int pixel_pair_in_fern,
        int first_level_num,
        int second_level_num){
    cout<<"Start training..."<<endl;
    vector<Mat_<double> > augment_target_shapes;
    vector<Mat_<uchar> > images; 
    vector<Mat_<double> > augment_current_shapes; 
    RNG random_generator(getTickCount());
    for(int i = 0;i < input_images.size();i++){
        Mat_<uchar> temp = input_images[i].clone(); 
        for(int j = 0;j < initial_number;j++){
            images.push_back(temp);
            augment_target_shapes.push_back(target_shapes[i]);
            int index = 0;
            do{
                index = random_generator.uniform(0,input_images.size()); 
            }while(index == i);
            augment_current_shapes.push_back(mean_shape);
        }
    }

    ShapeRegressor regressor(mean_shape,images,augment_target_shapes,
            augment_current_shapes,first_level_num,
            second_level_num, pixel_pair_num,
            pixel_pair_in_fern);
    regressor.train();
    regressor.save("./data/model.txt");
}

Mat_<double> test(ShapeRegressor& regressor, const Mat_<uchar>& image, const vector<Mat_<double> > target_shapes,
        const Mat_<double>& mean_shape,
        int initial_number){
    RNG random_generator(getTickCount()); 
    Mat_<double> combine_shape;
    for(int i = 0;i < initial_number;i++){
        int index = 0;
        do{
            index = random_generator.uniform(0,target_shapes.size());  
        }while(index == i);
        // Mat_<double> shape = target_shapes[index].clone();
        Mat_<double> shape = mean_shape.clone();    
        regressor.predict(image,shape,mean_shape);
        if(i == 0){
            combine_shape = shape.clone();
        }else{
            combine_shape = combine_shape + shape;
        }
    }
    return combine_shape; 
}

// calculate the covariance of two vectors
// cov(x,y) = E((x - E(x)*(y-E(y))
double calculate_covariance(const vector<double>& v_1, const
        vector<double>& v_2){
    assert(v_1.size() == v_2.size());
    assert(v_1.size() != 0);
    double sum_1 = 0;
    double sum_2 = 0;
    double exp_1 = 0;
    double exp_2 = 0;
    double exp_3 = 0;
    for(int i = 0;i < v_1.size();i++){
        sum_1 += v_1[i];
        sum_2 += v_2[i];
    }
    exp_1 = sum_1 / v_1.size();
    exp_2 = sum_2 / v_2.size();
    for(int i = 0;i < v_1.size();i++){
        exp_3 = exp_3 + (v_1[i] - exp_1) * (v_2[i] - exp_2);
    }
    return exp_3 / v_1.size();
}


void show_image(const Mat_<uchar>& input_image, const Mat_<double>&  points){
    Mat_<uchar> image = input_image.clone();
    for(int i = 0;i < points.rows;i++){
        circle(image,Point2d(points(i,0),points(i,1)),3,Scalar(255,0,0),-1,8,0); 
    }
    imshow("image",image);
    // waitKey(2);
}