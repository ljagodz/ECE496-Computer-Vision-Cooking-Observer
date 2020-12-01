#include "ThermalVisionControl.h"

ThermalVisionControl::ThermalVisionControl(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    // Set Number of Captures to 0
    ui.CapNumDisplay->setText("0");

    // Initialize Out Directory Value
    std::string cwd = getcwd();
    ui.OutDirLine->setText(QString::fromStdString(cwd + "\\"));

    // Initialize Out Name Value
    ui.OutNameLine->setText(QString("sample_output_folder"));

    // Capture Display Loop
    connect(ui.SeekEnable, SIGNAL(clicked()), this, SLOT(UpdateSeek()));
    connect(ui.WebCamDisplayEnable, SIGNAL(clicked()), this, SLOT(UpdateWebCam()));

    // Capture-Interrupt Button
    connect(ui.SeekCapture, SIGNAL(clicked()), this, SLOT(UpdateSeek()));
    connect(ui.SeekCombo, QOverload<int>::of(&QComboBox::activated), [=](int index) { UpdateSeekDisplay(); });
    connect(ui.WebCamDisplay, SIGNAL(clicked()), this, SLOT(UpdateSeek()));

    connect(ui.CapStart, SIGNAL(clicked()), this, SLOT(StartCaptureData()));
    connect(ui.CapStop, SIGNAL(clicked()), this, SLOT(StopCaptureData()));

    // Open-Close Camera
    connect(ui.SeekOpen, SIGNAL(clicked()), this, SLOT(ToggleSeek()));
    connect(ui.WebCamOpen, SIGNAL(clicked()), this, SLOT(ToggleWebCam()));
}

void ThermalVisionControl::UpdateDisplay()
{
    if (ui.SeekEnable->isChecked())
    {
        UpdateSeek();
    }
    QTimer::singleShot(1000 / ui.SeekFrameRate->value(), this, SLOT(UpdateDisplay()));
}

void ThermalVisionControl::UpdateSeekDisplay()
{
    PrepareSeekDisplayImage();
    QImage SeekQImage((uchar*)SeekDisplayImage.data, SeekDisplayImage.cols, SeekDisplayImage.rows, SeekDisplayImage.step, QImage::Format_BGR888);
    ui.SeekDisplay->setPixmap(QPixmap::fromImage(SeekQImage));
}

bool ThermalVisionControl::UpdateSeekImage()
{
    if (!CaptureSeekTemperature())
    {
        return false;
    }
    ConvertSeekImage();
    return true;
}

void ThermalVisionControl::UpdateSeek()
{
    if (!UpdateSeekImage())
    {
        ui.SeekEnable->setCheckState(Qt::CheckState(false));
        return;
    }
    UpdateSeekDisplay();
    
    if (ui.SeekEnable->isChecked() || ui.SeekEnable->isTristate())
    {
        QTimer::singleShot(1000 / ui.SeekFrameRate->value(), this, SLOT(UpdateSeek()));
    }
}

bool ThermalVisionControl::UpdateWebCamImage()
{
    if (!CaptureWebCamImage())
    {
        return false;
    }
    RetrieveWebCamImage();
    return true;
}

void ThermalVisionControl::UpdateWebCamDisplay()
{
    PrepareWebCamDisplayImage(); 
    QImage WebCamQImage((uchar*)WebCamDisplayImage.data, WebCamDisplayImage.cols, WebCamDisplayImage.rows, WebCamDisplayImage.step, QImage::Format_BGR888);
    ui.WebCamDisplay->setPixmap(QPixmap::fromImage(WebCamQImage));
}

void ThermalVisionControl::UpdateWebCam()
{
    if (!UpdateWebCamImage())
    {
        ui.WebCamDisplayEnable->setCheckState(Qt::CheckState(false));
        return;
    }
    UpdateWebCamDisplay();

    if (ui.WebCamDisplayEnable->isChecked() || ui.WebCamDisplayEnable->isTristate())
    {
        QTimer::singleShot(1000 / ui.WebCamDisplayFrameRate->value(), this, SLOT(UpdateWebCam()));
    }
}


void ThermalVisionControl::ConvertSeekImage()
{
    t_mutex.lock();
    t_cam.convert(SeekTemperature, SeekImage);
    t_mutex.unlock(); 
}

bool ThermalVisionControl::CaptureSeekTemperature()
{
    if (!t_cam.isopen())
    {
        ui.statusBar->showMessage("Seek Camera is not opened.");
        return false;
    }
    t_mutex.lock();
    t_cam.capture(SeekTemperature);
    t_mutex.unlock();
    return true;
}

void ThermalVisionControl::ResizeSeekDisplayImage()
{
    t_mutex.lock();
    cv::resize(SeekImage, SeekDisplayImage, cv::Size(ui.SeekDisplay->width(), ui.SeekDisplay->height()), 0, 0, cv::INTER_LINEAR);
    t_mutex.unlock();
}

void ThermalVisionControl::ColorSeekDisplayImage()
{
    t_mutex.lock();
    // Apply ColorMap
    int colormap = ui.SeekCombo->currentIndex();
    switch (colormap)
    {
    case(TVC_COLORMAP_BLUE):
        cv::applyColorMap(SeekDisplayImage, SeekDisplayImage, cv::COLORMAP_WINTER);
        break;
    case(TVC_COLORMAP_RED):
        cv::applyColorMap(SeekDisplayImage, SeekDisplayImage, cv::COLORMAP_HOT);
        break;
    case(TVC_COLORMAP_RAINBOW):
        cv::applyColorMap(SeekDisplayImage, SeekDisplayImage, cv::COLORMAP_JET);
        break;
    case(TVC_COLORMAP_BW):
        cv::applyColorMap(SeekDisplayImage, SeekDisplayImage, cv::COLORMAP_BONE);
        break;
    }
    t_mutex.unlock();
}

bool ThermalVisionControl::CaptureWebCamImage()
{
    if (!w_cam.isopen())
    {
        ui.statusBar->showMessage("Web Camera is not opened.");
        return false; 
    }
    w_mutex.lock();
    w_cam.capture();
    w_mutex.unlock();
    return true;
}

void ThermalVisionControl::RetrieveWebCamImage()
{
    w_mutex.lock();
    w_cam.retrieve(WebCamImage);
    w_mutex.unlock();
}

void ThermalVisionControl::StartCaptureData()
{
    // Update UI
    ui.CapStart->setDisabled(true);
    ui.CapStop->setEnabled(true);

    // Loop Control
    i_mutex.lock();
    count = 1;
    iscapture = true;
    i_mutex.unlock();

    CaptureData();
}

void ThermalVisionControl::StopCaptureData()
{
    // Loop Control
    i_mutex.lock();
    count = 0;
    iscapture = false;
    i_mutex.unlock();

    // Update Display
    UpdateCountDisplay();

    // Update UI
    ui.CapStart->setEnabled(true);
    ui.CapStop->setDisabled(true);
}

void ThermalVisionControl::CaptureData()
{
    // Initialize New Directory
    fileOutputDir = QString(ui.OutDirLine->text() + ui.OutNameLine->text()).toStdString();
    UpdateFileSuffixes();

    // Create New Folder/Check if folder already/if folder already has data inside it 
    if (direxists(fileOutputDir))
    {
        int i = 1;
        while (direxists(fileOutputDir))
        {
            fileOutputDir = ui.OutDirLine->text().toStdString() + ui.OutNameLine->text().toStdString() + "_" + std::to_string(i);
            i++;
        }
        ui.statusBar->showMessage(QString::fromStdString("Output name already exists in this directory. Switch to: " + fileOutputDir));
    }

    if (!mkdir(fileOutputDir.c_str()))
    {
        ui.statusBar->showMessage(QString::fromStdString("Created directory: " + fileOutputDir));
    }
    else
    {
        ui.statusBar->showMessage(QString::fromStdString("Could not create directory: " + fileOutputDir));
        StopCaptureData();
        return;
    }

    // Set the camera settings
    fps = ui.CapFrameRate->value();
    widthSeekImg = ui.SeekWidth->value();
    heightSeekImg = ui.SeekHeight->value();
    widthWebCamImg = ui.WebWidth->value();
    heightWebCamImg = ui.WebHeight->value();

    // Enable Capture 
    CaptureDataEntry();
}

void ThermalVisionControl::CaptureDataEntry()
{
    // Update File Paths
    UpdateFilePaths();

    // Capture Images
    if (!CaptureWebCamImage())
    {
        StopCaptureData();
        return;
    }
    if (!CaptureSeekTemperature())
    {
        StopCaptureData();
        return;
    }

    // Retrieve Image Data
    ConvertSeekImage();
    RetrieveWebCamImage();

    // Convert to Desired Resolution    
    ResizeImages();

    // Display Images
    UpdateSeekDisplay();
    UpdateWebCamDisplay();
    UpdateCountDisplay();

    // Save image data to file 
    SaveImages();

    // Save temperature data to file (FILE TYPE?)

    // Save metadata to file
    
    // Loop Control
    i_mutex.lock();
    if (iscapture)
    {
        count++;
        QTimer::singleShot(1000 / fps, this, SLOT(CaptureDataEntry()));
    }
    i_mutex.unlock();
}

void ThermalVisionControl::UpdateFileSuffixes()
{
    suffixSeekImg = ui.SeekImgSuffix->text().toStdString();
    suffixSeekTemp = ui.SeekTempSuffix->text().toStdString();
    suffixWebCamImg = ui.WebImgSuffix->text().toStdString();
}

void ThermalVisionControl::UpdateFilePaths()
{
    fileSeekImg = fileOutputDir + "\\" + suffixSeekImg + "_" + std::to_string(count) + ".bmp";
    fileSeekTemp = fileOutputDir + "\\" + suffixSeekTemp + "_" + std::to_string(count) + ".csv";
    fileWebCamImg = fileOutputDir + "\\" + suffixWebCamImg + "_" + std::to_string(count) + ".bmp";
}

void ThermalVisionControl::ResizeImages()
{
    t_mutex.lock();
    w_mutex.lock();
    cv::resize(SeekImage, SeekSaveImage, cv::Size(widthSeekImg, heightSeekImg), 0, 0, cv::INTER_LINEAR);
    cv::resize(WebCamImage, WebCamSaveImage, cv::Size(widthWebCamImg, heightWebCamImg), 0, 0, cv::INTER_LINEAR);
    t_mutex.unlock();
    w_mutex.unlock();
}

void ThermalVisionControl::SaveImages()
{
    cv::imwrite(fileSeekImg, SeekSaveImage);
    cv::imwrite(fileWebCamImg, WebCamSaveImage);
}

void ThermalVisionControl::PrepareSeekDisplayImage()
{
    ResizeSeekDisplayImage();
    ColorSeekDisplayImage();
}

void ThermalVisionControl::PrepareWebCamDisplayImage()
{
    w_mutex.lock();
    cv::resize(WebCamImage, WebCamDisplayImage, cv::Size(ui.WebCamDisplay->width(), ui.WebCamDisplay->height()), 0, 0, cv::INTER_LINEAR);
    w_mutex.unlock();
}

void ThermalVisionControl::UpdateCountDisplay()
{
    i_mutex.lock();
    ui.CapNumDisplay->setText(QString::fromStdString(std::to_string(count)));
    i_mutex.unlock();
}

void ThermalVisionControl::ToggleSeek()
{
    t_mutex.lock();
    if (ui.SeekOpen->isChecked())
    {
        t_cam.open();
        if (t_cam.isopen())
        {
            ui.statusBar->showMessage("Seek Camera opened succesfully.");
        }
        else
        {
            ui.statusBar->showMessage("Seek Camera could not be opened.");
            ui.SeekOpen->toggle();
        }
    }
    else
    {
        t_cam.close();
        ui.statusBar->showMessage("Seek Camera closed.");
    }
    t_mutex.unlock();
}

void ThermalVisionControl::ToggleWebCam()
{
    w_mutex.lock();
    if (ui.WebCamOpen->isChecked())
    {
        w_cam.open();
        if (w_cam.isopen())
        {
            ui.statusBar->showMessage("Web Camera opened succesfully.");
        }
        else
        {
            ui.statusBar->showMessage("Web Camera could not be opened.");
            ui.WebCamOpen->toggle();
        }
    }
    else
    {
        w_cam.close();
        ui.statusBar->showMessage("Web Camera closed.");
    }
    w_mutex.unlock();
}