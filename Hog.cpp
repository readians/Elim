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

#define PI 3.14159265

Mat get_hogdescriptor_visual_image(Mat& origImg,
                                   vector<float>& descriptorValues,
                                   Size winSize,
                                   Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor)
{   
    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
 
    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = (float)3.14/(float)gradientBinSize; 
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
	
	/*Without overlap*/
	 for (cellx=0; cellx<cells_in_x_dir; cellx++)
    {
        for (celly=0; celly<cells_in_y_dir; celly++)            
        {
			for (int bin=0; bin<gradientBinSize; bin++)
            {
                float gradientStrength = descriptorValues[ descriptorDataIdx ];
                descriptorDataIdx++;
 
                gradientStrengths[celly][cellx][bin] += gradientStrength;
            } 
            cellUpdateCounter[celly][cellx]++;
        }
    }

    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
 
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
 
            rectangle(visual_image,
                      Point(drawX*scaleFactor,drawY*scaleFactor),
                      Point((drawX+cellSize.width)*scaleFactor,
                      (drawY+cellSize.height)*scaleFactor),
                      CV_RGB(100,100,100),
                      1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = (float) cellSize.width/2;
                float scale = (float)viz_factor; // just a visual_imagealization scale,
                                          // to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visual_imagealization
                line(visual_image,
                     Point((int)x1*scaleFactor,(int)y1*scaleFactor),
                     Point((int)x2*scaleFactor,(int)y2*scaleFactor),
                     CV_RGB(0,0,255),
                     1);
            } // for (all bins)
        } // for (cellx)
    } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
		for (int x=0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];            
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return visual_image;
}

int main( int argc, char** argv )
{
	/*-------------------------------------------------------------------------------------*/
	VideoCapture cap(0); // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame, curr;
	Mat hog;
	vector<float> descriptorsValues;
	vector<Point> locations;
	int cellsize = 8;
	HOGDescriptor d(Size(640,480),Size(cellsize,cellsize),Size(cellsize,cellsize),Size(cellsize,cellsize),9,0,-1,HOGDescriptor::L2Hys, 0.2, false, HOGDescriptor::DEFAULT_NLEVELS);
				// Size(640,480), //winSize
				// Size(16,16), //blocksize
				// Size(16,16), //blockStride, if equal to blocksize there is no overlap
				// Size(16,16), //cellSize
				// 9, //nbins,
				// 0, //derivAper,
				// -1, //winSigma,
				// 0, //histogramNormalizationType,
				// 0.2, //L2HysThresh,
				// false //gamma correction,
				// nlevels=64
	int WIDTH,HEIGHT;
	int i,j,k;
	int patch_x;
	int patch_y;
	double **C;
	double p,s;
	/************************************************************************************************************/

	cap >> frame;

	WIDTH = frame.cols; //640 on hp-pavillion webcam
	HEIGHT = frame.rows; //480 on hp-pavillion webcam
	patch_x = WIDTH/cellsize; //40 on hp-pavillion webcam
	patch_y = HEIGHT/cellsize; //30 on hp-pavillion webcam

	C = new double*[patch_y];
	for(i = 0; i < patch_y; ++i)
		C[i] = new double[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			C[i][j] = 0;

	Mat Cs = Mat::zeros(HEIGHT*3,WIDTH*3,frame.type());

	/*Start*/
	while(1)
    {
		cvtColor(frame, gray_frame, CV_RGB2GRAY);
		
		gray_frame.convertTo(curr, -1, 1.2, 0);
		
		d.compute(curr, descriptorsValues, Size(0,0), Size(0,0), locations);
		
		hog = get_hogdescriptor_visual_image(curr,descriptorsValues,Size(WIDTH,HEIGHT),Size(cellsize,cellsize),3,2);

		for(i=0;i<patch_y;i++)
			{
				for(j=0;j<patch_x;j++)
				{
					s = 0;
					for(k=0;k<9;k++)
						s += descriptorsValues[(j * patch_y + i) * 9 + k];
					for(k=0;k<9;k++)
					{
						p = descriptorsValues[(j * patch_y + i) * 9 + k]/s;
						C[i][j] += p*log10(p);
					}
					C[i][j] = -C[i][j];
				}
			}

		for (int y = 0; y < Cs.rows; y+=cellsize*3)
			for (int x = 0; x < Cs.cols; x+=cellsize*3)
			{
				i = y/(cellsize*3);
				j = x/(cellsize*3);
				ostringstream strs;
				strs << C[i][j];
				string str = strs.str();
				str = str.substr(0,4);
				rectangle(Cs, Point(x,y), Point(x+cellsize*3,y+cellsize*3), 127, 1);
				putText(Cs, str, cvPoint(x,y+(cellsize)), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255,255,0));
			}
		circle(curr, Point((39*cellsize+cellsize/2),(1*cellsize)+cellsize/2), 7, 127, 2, 8, 0);

		imshow("Hog",hog);
		imshow("frame",Cs);
		
		Cs = Mat::zeros(HEIGHT*3,WIDTH*3,frame.type());
		for(i=0;i<patch_y;i++)
			for(j=0;j<patch_x;j++)
				C[i][j] = 0;

		cap >> frame; // read a new frame from video
		
		if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
		{
			system("PAUSE");
			cout << "esc key is pressed by user" << endl; 
            break; 
		}
    }
}
