@echo off

if %1.==. goto No1

set RootPath=%1
set PYTHONPATH=%RootPath%\models;%RootPath%\models\research;%RootPath%\models\research\slim;%RootPath%\models\research\slim\datasets;%RootPath%\models\research\slim\deployment;%RootPath%\models\research\slim\nets;%RootPath%\models\research\slim\preprocessing;%RootPath%\models\research\slim\scripts;%RootPath%\models\research\object_detection
set PATH=%PATH%;%PYTHONPATH%
echo Paths added to PYTHONPATH and PATH

goto End1

:No1
  echo.
  echo Usage: %~n0 PROJECT_PATH
  echo E.g.: %~n0 c:\DeepIGT
  echo.
goto End1

:End1
