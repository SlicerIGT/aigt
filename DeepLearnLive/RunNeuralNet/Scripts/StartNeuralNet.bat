set CurrentPath=%1
set NetworkType=%2
set ModelPath=%3
set ModelName=%4
set OutputType=%5
set IncomingHostName=%6
set IncomingPort=%7
set OutgoingHostName=%8
set OutgoingPort=%9
shift
set DeviceName=%9

set EnvironmentPath=kerasGPUEnv
call conda activate %EnvironmentPath%
call python %CurrentPath%/kerasNeuralNetwork.py --network_module_name=%NetworkType% --model_name=%ModelName% --model_directory=%ModelPath% --output_type=%OutputType% --incoming_host=%IncomingHostName% --incoming_port=%IncomingPort% --outgoing_host=%OutgoingHostName% --outgoing_port=%OutgoingPort% --device_name=%DeviceName%
call conda deactivate
::call conda activate kerasGPUEnv
::call python C:/Users/hisey/Documents/DeepLearnLive/RunNeuralNet/Scripts/kerasNeuralNetwork.py --network_module_name=CNN_LSTM --model_name=TBME_0 --model_directory=c:/Users/hisey/Documents/DeepLearnLive/Networks/CNN_LSTM/ --output_type=STRING --incoming_host=localhost --incoming_port=18945 --outgoing_host=localhost --outgoing_port=18944
::call conda deactivate