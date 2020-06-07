@ECHO OFF

 

IF %1.==. GOTO No1

 

set INPUT_FILE=%1

set OUTPUT_FILE=%~d1%~p1%~n1_TrackerToNeedle.mha

set CONFIG_FILE=%~d1%~p1%~n1_config.xml

set OUTPUT_CONFIG_FILE=%~d1%~p1%~n1_TrackerToNeedle_config.xml

 

EditSequencefile --operation=ADD_TRANSFORM --add-transform=TrackerToNeedle --source-seq-file=%INPUT_FILE% --output-seq-file=%OUTPUT_FILE% --config-file=%CONFIG_FILE% --use-compression

 

copy %CONFIG_FILE% %OUTPUT_CONFIG_FILE%

 

GOTO End1

 

:No1

  ECHO.

  ECHO Usage: %~n0 YourRecordedSequence.mha

  ECHO.

GOTO End1

 

:End1