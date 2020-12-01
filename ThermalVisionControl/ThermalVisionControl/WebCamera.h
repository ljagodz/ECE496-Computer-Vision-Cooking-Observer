#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class WebCamera
{
public:
	WebCamera();
	~WebCamera(); 

	bool open() { return m_cam.open(0); };
	bool isopen() { return m_cam.isOpened(); };

	void close() { return m_cam.release(); };
	
	bool capture() { return m_cam.grab(); };
	bool retrieve(cv::Mat& img) { return m_cam.retrieve(img); }

	int getwidth() { return m_cam.get(cv::CAP_PROP_FRAME_WIDTH); }
	int getheight() { return m_cam.get(cv::CAP_PROP_FRAME_HEIGHT);}
private:
	cv::VideoCapture m_cam;
};