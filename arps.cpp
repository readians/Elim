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

float costFunMAD(Mat curr, Mat ref, int n){
	int i,j;
	float err = 0;

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			err = err + abs(curr.data[curr.cols*i + j] - ref.data[ref.cols*i + j]);

	return err / (n*n);
}

float min_arr(float *a){
	float min = 65537;

	for(int i=0;i<sizeof(a)/sizeof(int);i++)
		if (a[i] < min)
			min = a[i];

	return min;
}

int min_arr_point(float *a){
	int point = 0;
	float min = a[0];

	for(int i=0;i<sizeof(a)/sizeof(int);i++)
		if (a[i] < min)
			point = i;
	
	return point;
}

int **BMA(Mat A, Mat B, int blocksize, int p)
{
	//Mat flow = Mat::zeros(A.rows/blocksize,A.cols/blocksize, A.type());
	int **checkMatrix;
	int WIDTH = A.cols;
	int HEIGHT = A.rows;
	int **MV;
	int refBlkVer,refBlkHor;
	int i,j,k,l,x,y,stepSize,maxIndex,doneFlag;
	int SDSP[5][2];
	int LDSP[6][2];
	int mbCount,point;
	float minMAD = 255;
	float costs[5] = {65537,65537,65537,65537,65537};
	float cost;

	checkMatrix = new int*[(2*p)+1];
	for(i = 0; i < (2*p)+1; ++i)
		checkMatrix[i] = new int[(2*p)+1];
	for(i=0;i<(2*p)+1;i++)
		for(j=0;j<(2*p)+1;j++)
			checkMatrix[i][j] = 0;

	MV = new int*[2];
	for(i = 0; i < 2; ++i)
		MV[i] = new int[(A.cols*A.rows)/(blocksize^2)];
	for(i=0;i<2;i++)
		for(j=0;j<(A.cols*A.rows)/(blocksize^2);j++)
			MV[i][j] = 0;

	SDSP[0][0] = 0; SDSP[0][1] = -1;
	SDSP[1][0] = -1; SDSP[1][1] = 0;
	SDSP[2][0] = 0; SDSP[2][1] = 0;
	SDSP[3][0] = 1; SDSP[3][1] = 0;
	SDSP[4][0] = 0; SDSP[4][1] = 1;

	/*Start computing motion vectors for each block of the frame*/
	mbCount = 0;
	for(i=0;i<=HEIGHT-blocksize;i+=blocksize)
		for(j=0;j<=WIDTH-blocksize;j+=blocksize)
		{
			x = j;
			y = i;

			Mat currblck(A, Rect(i, j, blocksize, blocksize));
			Mat refblck(B, Rect(i, j, blocksize, blocksize));
			costs[2] = costFunMAD(currblck,refblck,blocksize);
			checkMatrix[p+1][p+1] = 1;

			if (j-1 < 0)
			{
				stepSize = 2;
				maxIndex = 5;
			}
			else 
			{
				stepSize = max(abs(MV[0][mbCount-1]), abs(MV[1][mbCount-1]));
				if ((abs(MV[0][mbCount-1]) == stepSize && MV[1][mbCount-1] == 0)
					|| (abs(MV[1][mbCount-1]) == stepSize && (MV[0][mbCount-1] == 0)))
					maxIndex = 5; 
				else
				{
					maxIndex = 6;
					LDSP[5][0] =  MV[1][mbCount-1]; LDSP[5][1] = MV[0][mbCount-1];
				}
			}

			LDSP[0][0] = 0; LDSP[0][1] = -stepSize;
			LDSP[1][0] = -stepSize; LDSP[1][1] = 0;
			LDSP[2][0] = 0; LDSP[2][1] = 0;
			LDSP[3][0] = stepSize; LDSP[3][1] = 0;
			LDSP[4][0] = 0; LDSP[4][1] = stepSize;

			for(k=0;k<maxIndex;k++)
			{
				refBlkVer = y + LDSP[k][1];   
				refBlkHor = x + LDSP[k][0];  
				if ( refBlkVer < 0 || refBlkVer+blocksize-1 >= HEIGHT || refBlkHor < 0 || refBlkHor+blocksize-1 >= WIDTH)
					continue;
			
				if (k == 3 || stepSize == 0)
					continue; 
				
				Mat currblck(A, Rect(i, j, blocksize, blocksize));
				Mat refblck(B, Rect(refBlkVer, refBlkHor, blocksize, blocksize));
				costs[k] = costFunMAD(currblck,refblck,blocksize);
				checkMatrix[LDSP[k][1] + p+1][LDSP[k][0] + p+1] = 1;
			}

			cost = min_arr(costs);
			point = min_arr_point(costs);

			x = x + LDSP[point][0];
			y = y + LDSP[point][1];

			for(l=0;l<5;l++) costs[l] = 65537;
			costs[2] = cost;

			doneFlag = 0;   
			while (doneFlag == 0)
			{
				for(k=0;k<5;k++)
				{
					refBlkVer = y + SDSP[k][1]; 
					refBlkHor = x + SDSP[k][0];

					if ( refBlkVer < 0 || refBlkVer+blocksize-1 >= HEIGHT || refBlkHor < 0 || refBlkHor+blocksize-1 >= WIDTH)
						  continue;
					
					if (k == 2)
						continue;
					else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p || refBlkVer > i+p)
							continue;
					else if (checkMatrix[y-i+SDSP[k][2]+p+1][x-j+SDSP[k][1]+p+1] == 1)
						continue;
					
					Mat currblck(A, Rect(i, j, blocksize, blocksize));
					Mat refblck(B, Rect(refBlkVer, refBlkHor, blocksize, blocksize));
					costs[k] = costFunMAD(currblck,refblck,blocksize);
					
					checkMatrix[y-i+SDSP[k][2]+p+1][x-j+SDSP[k][1]+p+1] = 1;
				}
            
				cost = min_arr(costs);
				point = min_arr_point(costs);
           
				if (point == 2)
					doneFlag = 1;
				else
				{
					x = x + SDSP[point][0];
					y = y + SDSP[point][1];
					for(l=0;l<5;l++) costs[l] = 65537;
					costs[2] = cost;
				}
			}

			MV[0][mbCount] = y - i;
			MV[1][mbCount] = x - j;              
			mbCount = mbCount + 1;
			for(l=0;l<5;l++) costs[l] = 65537;
        
			for(i=0;i<(2*p)+1;i++)
				for(j=0;j<(2*p)+1;j++)
					checkMatrix[i][j] = 0;
		}
	return MV;
}


int main( int argc, char** argv )
{
	VideoCapture cap; // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame;
	Mat prev,next;
	Mat bgr;//CV_32FC3 matrix
	Mat flow;
	Mat SM;
	vector<float> descriptorsValues;
	vector<Point> locations;
	HOGDescriptor d(Size(640,480),Size(16,16),Size(16,16),Size(16,16),9,0,-1,HOGDescriptor::L2Hys, 0.2, false, HOGDescriptor::DEFAULT_NLEVELS);
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
	int i,j,k,N=0;	
	int WIDTH,HEIGHT;
	int patch_x;
	int patch_y;
	int maxLevel=3;
	int flags = 0;
	int **MV;
	float ***p;
	float **C;
	float s;

	//namedWindow("Coherency Based Spatio-Temporal SM (Up to 15 frames)", CV_WINDOW_AUTOSIZE);
	namedWindow("Optical Flow",1);
	
	cap.open(0);
	
	WIDTH = 640; //640 on hp-pavillion webcam
	HEIGHT = 480; //480 on hp-pavillion webcam
	patch_x = WIDTH/16; //40 on hp-pavillion webcam
	patch_y = HEIGHT/16; //30 on hp-pavillion webcam
	
	C = new float*[patch_y];
	for(i = 0; i < patch_y; ++i)
		C[i] = new float[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			C[i][j] = 0;
	p = new float**[patch_y];
	for(i = 0; i < patch_y; i++)
	{
		p[i] = new float*[patch_x];
		for(j = 0; j < patch_x; j++)
		{
			p[i][j] = new float[9];
		}
	}
	for(i = 0; i < patch_y; i++)
		for(j = 0; j < patch_x; j++)
			for(k=0;k<9;k++)
				p[i][j][k] = 0;

	MV = new int*[2];
	for(i = 0; i < 2; ++i)
		MV[i] = new int[(WIDTH*HEIGHT)/(16^2)];

	while(1)
    {
		//for(N=0;N<15;N++)
		//{
			cap >> frame; // read a new frame from video			
			
			cvtColor(frame, gray_frame, CV_BGR2GRAY);

			gray_frame.convertTo(next, -1, 1.2, 0);
			
			if(N > 0)
			{
				MV = BMA(prev, next, 16, 7);
				imshow("Optical Flow", frame);
			}
			N = 1;
			swap(prev, next);
			
			if(N == 14)
			{
				d.compute(next, descriptorsValues, Size(0,0), Size(0,0), locations);
		
				for(i=0;i<patch_x;i++)
				{
					for(j=0;j<patch_y;j++)
					{
						s = 0;
						for(k=0;k<9;k++)
							s += descriptorsValues[(i * patch_y + j) * 9 + k];
						for(k=0;k<9;k++)
						{
							p[j][i][k] = descriptorsValues[(i * patch_y + j) * 9 + k]/s;
							C[j][i] += p[j][i][k]*log10(p[j][i][k]);
						}
						C[j][i] = -C[j][i];
					}
				}
			}
		//}
		
		//SM = formula della mappa SM;

		//imshow("Coherency Based Spatio-Temporal SM (Up to 15 frames)",SM);

		for(i=0;i<patch_y;i++)
				for(j=0;j<patch_x;j++)
					C[i][j] = 0;
		
		if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
		{
			system("PAUSE");
			cout << "esc key is pressed by user" << endl; 
            break; 
		}
    }
}
