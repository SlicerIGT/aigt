set CondaPath=%1
Echo %CondaPath%
set EnvName=%2
Echo %EnvName%
set DataCSV=%3
set SaveLocation=%4
set TrainingScript=%5

call %CondaPath%\Scripts\activate %EnvName%
call python %TrainingScript% --save_location=%SaveLocation% --data_csv_file=%DataCSV% 
call %CondaPath%\condabin\conda.bat deactivate