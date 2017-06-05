#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <SFML/Audio.hpp>
#include <iomanip>
void playSound(int);

using namespace cv;
using namespace std;

// Global Variables definition
int pre_stop=0, pre_ped=0,pre_school=0,now_stop=0,now_ped=0,now_school=0;
char* source_window = "Source";
Mat image;

//Start of Main Code
int main( int argc, char** argv )
{	VideoCapture cap(0);										// Video Frame capturing structure
	if(!cap.isOpened())											//If not open then return
    return -1;

	// Training code of System for three road signs 
    FileStorage fs1,fs2,fs3;
    fs1.open("SVMpd.xml", FileStorage::READ);					//Reading training data for Pedestrain sign
	fs2.open("SVMst.xml", FileStorage::READ);					//Reading training data for Stop sign
	fs3.open("SVMsc.xml", FileStorage::READ);					//Reading training data for School sign
    Mat SVM_TrainingData1,SVM_TrainingData2,SVM_TrainingData3;
    Mat SVM_Classes1,SVM_Classes2,SVM_Classes3;
    fs1["TrainingData"] >> SVM_TrainingData1;
	fs2["TrainingData"] >> SVM_TrainingData2;
	fs3["TrainingData"] >> SVM_TrainingData3;
    fs1["classes"] >> SVM_Classes1;
	fs2["classes"] >> SVM_Classes2;
	fs3["classes"] >> SVM_Classes3;
    
    CvSVMParams SVM_params;										// Setting the parameters of SVM
    SVM_params.svm_type = CvSVM::C_SVC;							// Following lines set different
    SVM_params.kernel_type = CvSVM::LINEAR;						// parameters related to linear type 
    SVM_params.degree = 0;										// type of support vector machine
    SVM_params.gamma = 1;
    SVM_params.coef0 = 0;
    SVM_params.C = 1;
    SVM_params.nu = 0;
    SVM_params.p = 0;											// Following lines get classifiers for ROI
    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
    CvSVM svmClassifier1(SVM_TrainingData1, SVM_Classes1, Mat(), Mat(), SVM_params);
	CvSVM svmClassifier2(SVM_TrainingData2, SVM_Classes2, Mat(), Mat(), SVM_params);
	CvSVM svmClassifier3(SVM_TrainingData3, SVM_Classes3, Mat(), Mat(), SVM_params);
	cv::namedWindow("Image");

 //Loop starts here that processess frames
while(1)
{
	cap >> image;												// Get image frame from camera
	cv::cvtColor(image,image,CV_BGR2GRAY);						// Convert the image to gray scale
	GaussianBlur(image, image, Size(7,7), 1.5, 1.5);			// Blur the image to remove noise
	cv::Mat edges;
	Canny(image, edges,0, 30);									// Getting Edges of image
	cv::imshow("Image",edges);
	waitKey(30);
	cout << "Analysing the Image....!!" << endl;				//Display message
	
	//Now i am finding Contours for the scene that i got
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges,	contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); // Drawing contours along edges

	int cmin= 300; 
	std::vector<std::vector<cv::Point>>::const_iterator itc= contours.begin();
	int i=0;
	while (itc!=contours.end()) {
		double area = contourArea(contours[i]);					// Here are valid contour checks
		cv::Rect r0= cv::boundingRect(cv::Mat(contours[i]));
		double extent = area/(r0.width*r0.height);				// Checking for irregular contour
				
				if(r0.height > (r0.width + 100) || r0.width > (r0.height +100))
					itc= contours.erase(itc);
				else if(extent < 0.6)
				itc= contours.erase(itc);						//Discard abnormal contour
				else
				{
					Mat crop = image(r0);
					resize(crop,crop,cv::Size(60,60));			//Resizing the crop to make
					cv::imshow("Image",crop);					// it standard in all cases
					waitKey(30);
                    Mat p= crop.reshape(1, 1);
					p.convertTo(p, CV_32FC1);

					int response1 = (int)svmClassifier1.predict( p ); // Getting match results from database
					int response2 = (int)svmClassifier2.predict( p );
					int response3 = (int)svmClassifier3.predict( p );
					if(response1==1 && now_ped<1 && pre_ped<1)			 // If Any of three Signs found then
					{cout<<"Padestrian"<<endl; now_ped+=1; playSound(1);}// display the message and play sound
					else if(response2==1 && now_stop<1 && pre_stop<1)
					{cout<<"Stop"<<endl; playSound(2); now_stop+=1; }
					else if(response3==1 && now_school<1 && pre_school<1)
					{cout<<"School"<<endl; playSound(3); now_school+=1;}
					else cout<<"No match of ROI found"<<endl;
					++itc; response1 = 0; response2 = 0; response3 = 0;
					i++;
				}
		
	}
	pre_ped = now_ped; pre_school = now_school; pre_stop = now_stop;		//Reset some parameters to start
	now_school = 0; now_stop = 0; now_ped = 0;
	waitKey(500);
}
      return 0;
}

void playSound(int a)
{
    sf::SoundBuffer buffer;						//Load buffer with audio file
	if(a==1)
	{buffer.loadFromFile("pd.wav");}			//Audios related to each sign
	if(a==2)
	{buffer.loadFromFile("st.wav");}
	if(a==3)
	{buffer.loadFromFile("sc.wav");}
    sf::Sound sound(buffer);
    sound.play();
	    while (sound.getStatus() == sf::Sound::Playing) // Soind playing algorithm
    {
        sf::sleep(sf::milliseconds(100));
    }
return;
}