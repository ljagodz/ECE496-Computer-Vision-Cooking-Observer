#include "ThermalVisionControl.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ThermalVisionControl w;
    w.show();
    return a.exec();
}
