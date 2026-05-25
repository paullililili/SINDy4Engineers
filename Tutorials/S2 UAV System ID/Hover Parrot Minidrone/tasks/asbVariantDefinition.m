%% asbVariantDefinition.m - Variants definiton
% This script initializes the VariantExpression objects for the quadcopter model.

% Copyright 2017-2019 The MathWorks, Inc.

% Command
VSS_COMMAND_SIGBLDR = Simulink.VariantExpression('VSS_COMMAND==0');
VSS_COMMAND_JOYSTICK = Simulink.VariantExpression('VSS_COMMAND==1');
VSS_COMMAND_PRESAVED = Simulink.VariantExpression('VSS_COMMAND==2');
VSS_COMMAND_SPREADSHEET = Simulink.VariantExpression('VSS_COMMAND==3');

% Sensors
VSS_SENSORS_FEEDTHROUGH = Simulink.VariantExpression('VSS_SENSORS==0');
VSS_SENSORS_DYNAMICS = Simulink.VariantExpression('VSS_SENSORS==1');

% Environment
VSS_ENVIRONMENT_CST = Simulink.VariantExpression('VSS_ENVIRONMENT==0');
VSS_ENVIRONMENT_VARIABLE = Simulink.VariantExpression('VSS_ENVIRONMENT==1');

% Vehicle
VSS_VEHICLE_LINEAR = Simulink.VariantExpression('VSS_VEHICLE==0');
VSS_VEHICLE_NONLINEAR = Simulink.VariantExpression('VSS_VEHICLE==1');
