set CondaPath=%1
Echo %CondaPath%
set EnvName=%2
Echo %EnvName%
set CurrentPath=%3
set NetworkType=%4
set ModelPath=%5
set ModelName=%6
set OutputType=%7
set IncomingHostName=%8
set IncomingPort=%9
shift
set OutgoingHostName=%9
shift
set OutgoingPort=%9
shift
set DeviceName=%9

call %CondaPath%\Scripts\activate %EnvName%
call python %CurrentPath%\kerasNeuralNetwork.py --network_module_name=%NetworkType% --model_name=%ModelName% --model_directory=%ModelPath% --output_type=%OutputType% --incoming_host=%IncomingHostName% --incoming_port=%IncomingPort% --outgoing_host=%OutgoingHostName% --outgoing_port=%OutgoingPort% --device_name=%DeviceName%
call %CondaPath%\condabin\conda.bat deactivate

if %ERRORLEVEL% neq 0 goto AlternateCondaDeactivate

:AlternateCondaDeactivate
call %CondaPath%\Scripts\deactivate