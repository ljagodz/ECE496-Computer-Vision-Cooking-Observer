#pragma once

#include <string>
#include <limits.h>
#include <fstream>
#include <iostream>
#include <direct.h>
#include <QtCore/qtimer.h>
#include <windows.h>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>
#include <QtWidgets/QMainWindow>
#include <QtCore/qmutex.h>
#include "WebCamera.h"
#include "ThermalCamera.h"
#include "ui_ThermalVisionControl.h"

#define TVC_COLORMAP_BLUE 0
#define TVC_COLORMAP_RED 1
#define TVC_COLORMAP_RAINBOW 2
#define TVC_COLORMAP_BW 3

class ThermalVisionControl : public QMainWindow
{
    typedef struct {
        
    } CaptureData;

    Q_OBJECT
public slots:
    void UpdateDisplay(); 

    void UpdateSeek();
    bool UpdateSeekImage();
    void UpdateSeekDisplay(); 

    void UpdateWebCam();
    bool UpdateWebCamImage();
    void UpdateWebCamDisplay();

    void UpdateCountDisplay();

    void StartCaptureData();
    void StopCaptureData();
    void CaptureDataEntry();

    void ToggleSeek();
    void ToggleWebCam();

public:
    ThermalVisionControl(QWidget* parent = Q_NULLPTR);

    bool CaptureSeekTemperature();
    void ConvertSeekImage();

    void ColorSeekDisplayImage();
    void ResizeSeekDisplayImage();

    bool CaptureWebCamImage();
    void RetrieveWebCamImage();

    void PrepareSeekDisplayImage();
    void PrepareWebCamDisplayImage();

    void ResizeImages();
    void SaveImages();
    void SaveCSV(std::string csvPath, cv::Mat csvMat);

    void UpdateFilePaths();
    void UpdateFileSuffixes();

    void CaptureData();

    std::string getcwd()
    {
        char buff[MAX_PATH];
        GetModuleFileName(NULL, buff, MAX_PATH);
        std::string::size_type position = std::string(buff).find_last_of("\\/");
        return std::string(buff).substr(0, position);
    }

    bool direxists(std::string pathname)
    {
        struct stat buffer;
        return (stat(pathname.c_str(), &buffer) == 0);
    }

private:
    WebCamera w_cam;
    ThermalCamera t_cam;
    Ui::ThermalVisionControlClass ui;
    
    // Frames
    cv::Mat SeekImage;
    cv::Mat SeekSaveImage;
    cv::Mat SeekDisplayImage;
    cv::Mat SeekTemperature;

    cv::Mat WebCamImage;
    cv::Mat WebCamSaveImage;
    cv::Mat WebCamDisplayImage;

    // Data File Paths
    std::string fileWebCamImg;
    std::string suffixWebCamImg;
    std::string fileSeekTemp, fileSeekImg;
    std::string suffixSeekTemp, suffixSeekImg;

    // Output File Paths
    std::string fileOutputDir; 

    // Resolutions
    int widthSeekImg, heightSeekImg;
    int widthSeekTemp, heightSeekTemp;
    int widthWebCamImg, heightWebCamImg;

    // Mutexes
    QMutex t_mutex;
    QMutex w_mutex;
    QMutex i_mutex;

    // Frame Rate
    double fps;

    // Loop Control
    int count; 
    bool iscapture; 
};
