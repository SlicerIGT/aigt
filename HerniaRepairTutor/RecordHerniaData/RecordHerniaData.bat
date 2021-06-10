title Record Hernia Data

START /MIN C:\Users\hisey\PlusApp-2.7.0.20190123-Telemed-Win32\bin\PlusServerLauncher.exe --connect --device-set-configuration-dir=c:\Users\hisey\Documents\Github\UsNeedleTutor\Config  --config-file=Telemed-50mm-L12_ImageToProbeAnalysis.xml

cd "C:\Users\hisey\AppData\Local\NA-MIC\Slicer 4.13.0-2021-02-03\"
START Slicer.exe --python-code "slicer.util.mainWindow().moduleSelector().selectModule('RecordHerniaData')"

