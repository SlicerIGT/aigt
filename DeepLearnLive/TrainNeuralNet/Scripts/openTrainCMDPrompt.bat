set CurrentPath=%1
set CondaPath=%2
Echo %CondaPath%
set EnvName=%3
Echo %EnvName%
set DataCSV=%4
set SaveLocation=%5
set TrainingScript=%6

call cmd.exe /K %CurrentPath%\Scripts\TrainNeuralNet.bat %CondaPath% %EnvName% %DataCSV% %SaveLocation% %TrainingScript%
