#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <sstream>
#include <string.h>
#include <math.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
	string arg = "1.JPG";
	Mat A,B;

	cout << "file path: " << arg << endl;
    A = imread(arg, CV_LOAD_IMAGE_COLOR); // Read the file
	if(!A.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << endl ;
        return -1;
    }
	cvtColor(A, B, CV_BGR2GRAY);

	for(int i=0; i<B.rows; i++){
	    for(int j=0; j<B.cols; j++){
			   B.data[B.cols*i + j] *= 1.3;
			   //otherwise...
			   //B.at<type>(i,j) *= 1.3;
			   /*for multichannel pics
			   //cout<<(int)A.data[A.step[0]*i + A.step[1]* j + 0]<<"-";
			   //cout<<(int)A.data[A.step[0]*i + A.step[1]* j + 1]<<"-";
			   //cout<<(int)A.data[A.step[0]*i + A.step[1]* j + 2]<<" ";*/
		}
	}
	namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", B ); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
