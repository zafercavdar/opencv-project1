
/*This script has test cases to help you test myFilter() which you will
write.You should verify that you get reasonable output here before using
your filtering to construct a hybrid image in proj1.cpp.The outputs are
all saved and you can include them in your writeup. You can add calls to
filter2D() if you want to check that myFilter() is doing something
similar.*/

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat myFilter(Mat, Mat, int);

Vec3d get_pixel_with_zero_padding(Mat im, int x, int y){
	if (x < 0 || y < 0)
		return Vec3d(0, 0, 0);
	else if(x >= im.cols || y >= im.rows)
		return Vec3d(0, 0, 0);
}

Vec3d get_pixel_with_replicate(Mat im, int x, int y){
	int w = im.cols;
	int h = im.rows;

	int rx;
	int ry;

	if (x < 0)
		rx = 0;
	else if (x >= w)
		rx = w - 1;
	
	if (y < 0)
		ry = 0;
	else if (y >= h)
		ry = h - 1;

	return im.at<Vec3d>(ry, rx);
}

Vec3d get_pixel_with_reflect(Mat im, int x, int y){
	int w = im.cols;
	int h = im.rows;

	int rx;
	int ry;

	if (x < 0)
		rx = -x;
	else if (x >= w)
		rx = (w - 1) * 2 - x;
	
	if (y < 0)
		ry = -y;
	else if (y >= h)
		ry = (h - 1) * 2 - y;
	
	return im.at<Vec3d>(ry, rx);
}


enum border { Border_Replicate, Border_Reflect, Border_Constant };

Mat myFilter(Mat im, Mat filter, int borderType = Border_Replicate)
{
	/*This function is intended to behave like the built in function filter2D()

	Your function should work for color images. Simply filter each color
	channel independently.

	Your function should work for filters of any width and height
	combination, as long as the width and height are odd(e.g. 1, 7, 9).This
	restriction makes it unambigious which pixel in the filter is the center
	pixel.

	Boundary handling can be tricky.The filter can't be centered on pixels
	at the image boundary without parts of the filter being out of bounds.
	There are several options to deal with boundaries. -- pad the input image with zeros, and
	return a filtered image which matches the input resolution. A better
	approach is to mirror the image content over the boundaries for padding.*/

	Mat outI = im.clone();
	int channels = im.channels();
	int row_boundary = filter.rows / 2;
	int column_boundary = filter.cols / 2;
	Scalar intensity;
	Vec3d color;

	for (int y= 0; y < outI.rows; y++){
		for (int x= 0; x < outI.cols; x++){
			if (channels == 3){

				// reset color accumulator
				color = Vec3d(0, 0, 0);
				
				// iterate over filter
				for (int j= 0; j < filter.rows; j++){
					for (int i= 0; i < filter.cols; i++){
						int index_j = j - filter.rows / 2;
						int index_i = i - filter.cols / 2;
						int target_x = x + index_i;
						int target_y = y + index_j;

						Vec3d pixel;
						// boundary
						if (target_x < 0 || target_y < 0 || target_x >= im.cols || target_y >= im.rows){
							switch(borderType){
								case Border_Constant:
									pixel = get_pixel_with_zero_padding(im, target_x, target_y);
									break;
								case Border_Replicate:
									pixel = get_pixel_with_replicate(im, target_x, target_y);
									break;
								case Border_Reflect:
									pixel = get_pixel_with_reflect(im, target_x, target_y);
									break;
								default:
									cout << "ERROR! borderType is not valid." << std::endl;
									pixel = Vec3d(0, 0, 0);
							}
						} else {
							pixel = im.at<Vec3d>(target_y, target_x);
						}

						//iterate over each channel
						for (int c=0; c < channels; c++){
							color[c] += pixel.val[c] * filter.at<double>(j, i);
						}
					}
				}
				// update output image with calculated color
				outI.at<Vec3d>(y, x) = color;

			} else if (channels == 1){
				// outI.at<uchar>(y, x) = 128;
			}
		}
	}
	return outI;
}

int main()
{
	//// Setup  ////
	//Load the test image
	Mat test_image = imread("data/cat.bmp");
	if (!test_image.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}	
	imshow("Test image", test_image);                   // Show the test image.
	waitKey(0);                                          // Wait for a keystroke in the window
	test_image.convertTo(test_image, CV_64FC3);

	//// Identify filter  ////
	//This filter should do nothing regardless of the padding method you use.
	Mat identity_filter = (Mat_<double>(3, 3) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
	//cout << identity_filter.at<double>(0,0); 
	Mat identity_image = myFilter(test_image, identity_filter);
	identity_image.convertTo(identity_image, CV_8UC3);
	imshow("Identity image", identity_image);
	waitKey(0);	
	imwrite("identity_image.jpg", identity_image); //save the identity image as jpeg


	////  Small blur with a box filter ////
	//This filter should remove some high frequencies
	Mat blur_filter = (Mat_<double>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	blur_filter = blur_filter / sum(blur_filter)[0]; //making the filter sum to 1
	Mat blur_image = myFilter(test_image, blur_filter);
	blur_image.convertTo(blur_image, CV_8UC3);
	imshow("Blur image", blur_image);
	waitKey(0);
	imwrite("blur_image.jpg", blur_image); //save the blur image as jpeg

	////   Large blur  ////
	//This blur would be slow to do directly, so we instead use the fact that
	//Gaussian blurs are separable and blur sequentially in each direction.
	Mat large_1d_blur_filter = getGaussianKernel(25, 10, CV_64F);
	Mat large_blur_image = myFilter(test_image, large_1d_blur_filter);
	large_blur_image = myFilter(large_blur_image, large_1d_blur_filter.t()); //notice the t() operator which transposes the filter
	large_blur_image.convertTo(large_blur_image, CV_8UC3);
	imshow("Large blur image", large_blur_image); waitKey(0);
	imwrite("large_blur_image.jpg", large_blur_image); //save the large blur image as jpeg
		
	////  Oriented filter(Sobel Operator)  ////
	Mat sobel_filter = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); //should respond to horizontal gradients
	Mat sobel_image = myFilter(test_image, sobel_filter);

	//the scalar value is added because the output image is centered around zero otherwise and mostly black
	sobel_image = sobel_image + Scalar(0.5, 0.5, 0.5) * 255;
	sobel_image.convertTo(sobel_image, CV_8UC3);
	imshow("Sobel image",sobel_image); waitKey(0);
	imwrite("sobel_image.jpg", sobel_image);

	////  High pass filter(Discrete Laplacian)   ////
	Mat laplacian_filter = (Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
	Mat laplacian_image = myFilter(test_image, laplacian_filter);
	//the scalar value is added because the output image is centered around zero otherwise and mostly black
	laplacian_image = laplacian_image + Scalar(0.5, 0.5, 0.5) * 255;
	laplacian_image.convertTo(laplacian_image, CV_8UC3);
	imshow("Laplacian image",laplacian_image); waitKey(0);
	imwrite("laplacian_image.jpg", laplacian_image);

	//// High pass "filter" alternative  ////
	blur_image.convertTo(blur_image, CV_64FC3);
	Mat high_pass_image = test_image - blur_image; //simply subtract the low frequency content
	
	//the scalar value is added because the output image is centered around zero otherwise and mostly black
	high_pass_image = high_pass_image + Scalar(0.5, 0.5, 0.5) * 255;
	high_pass_image.convertTo(high_pass_image, CV_8UC3);
	imshow("high pass image", high_pass_image); waitKey(0);
	imwrite("high_pass_image.jpg", high_pass_image);
}