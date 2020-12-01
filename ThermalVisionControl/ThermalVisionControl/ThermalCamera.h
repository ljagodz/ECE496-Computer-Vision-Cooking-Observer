#pragma once

#include <opencv2/highgui/highgui.hpp>
#include "seek.h"

class ThermalCamera
{
public:
	ThermalCamera();
	~ThermalCamera();

	bool open() { return m_seek.open(); };
	bool isopen() { return m_seek.isOpened(); };
	void close() { return m_seek.close(); };

	bool capture(cv::Mat& temp) { return m_seek.read(temp); };
	void convert(cv::Mat& temp, cv::Mat& img) { return m_seek.convertToGreyScale(temp, img); };

private:
	LibSeek::SeekThermal m_seek = LibSeek::SeekThermal(""); // Can add the FlatField Filename to load up a calibration 
};