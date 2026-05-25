clear;
clc;
close all;

% Set variable names
state_names = {'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Vx', 'Vy', 'Vz', 'p', 'q', 'r'};
input_names = {'u1', 'u2', 'u3', 'u4'};

% Load in signal template
load('Hover Parrot Minidrone/mainModels/Position_Attitude_Reference.mat');

% Close any existing projects
proj = matlab.project.rootProject;
if ~isempty(proj)
	close(proj);
end

% Open up Parrot project
proj = openProject('Hover Parrot Minidrone/HoverParrotMinidrone.prj');

%% Construct step input signals

% Set simulation time and refence command time points
tFinal = 150;

% Reset reference commands
Group_1 = new_signal(Group_1, tFinal);

% Create x steps
Group_1 = add_step(Group_1, 1, 10, 2);
Group_1 = add_step(Group_1, 1, 20, -2);

% Create y steps
Group_1 = add_step(Group_1, 2, 40, 2);
Group_1 = add_step(Group_1, 2, 50, -2);

% Create yaw steps
Group_1 = add_step(Group_1, 4, 70, pi/2);
Group_1 = add_step(Group_1, 4, 75, -pi/2);

% Create vertical steps
Group_1 = add_step(Group_1, 3, 90, -1);
Group_1 = add_step(Group_1, 3, 100, 1);

% Create coupled x-y steps
Group_1 = add_step(Group_1, 1, 110, 1);
Group_1 = add_step(Group_1, 2, 110, 1);

% Create coupled x-y-yaw steps
Group_1 = add_step(Group_1, 1, 130, -1);
Group_1 = add_step(Group_1, 2, 130, -1);
Group_1 = add_step(Group_1, 4, 130, pi/2);

% Write command to file
save('mainModels/Position_Attitude_Reference.mat', 'Group_1');

% Simulate flight
simout = sim('mainModels/parrotMinidroneHover.slx', tFinal);

% Visualise the resulting flight trajectory
visualise_sim(simout, Group_1, state_names, input_names);

% Extract and save data
save_sim('../data/Step Inputs.mat', simout);


%% Construct sine sweep chirp signal

% Set simulation time
tFinal = 90;

% Reset reference commands
Group_1 = new_signal(Group_1, tFinal);

% Add chirp signal in x
Group_1 = add_chirp(Group_1, 1, 10, 10, 0.1, 1, Ts, 10);

% Add chirl signal in y
Group_1 = add_chirp(Group_1, 2, 30, 10, 0.1, 1, Ts, 10);

% Add chirl signal in z
Group_1 = add_chirp(Group_1, 3, 50, 10, 0.1, 1, Ts, 1);

% Add chirl signal in yaw
Group_1 = add_chirp(Group_1, 4, 70, 10, 0.1, 1, Ts, 1);

% Write command to file
save('mainModels/Position_Attitude_Reference.mat', 'Group_1');

% Simulate flight
simout = sim('mainModels/parrotMinidroneHover.slx', tFinal);

% Visualise the resulting flight trajectory
visualise_sim(simout, Group_1, state_names, input_names);

% Extract and save data
save_sim('../data/Chirp Inputs.mat', simout);


%% Construct waypoint flight plan

% Set simulation time
tFinal = 90;

% Reset reference commands
Group_1 = new_signal(Group_1, tFinal);

% Add waypoint 1 (with mid-course altitude change)
Group_1 = add_step(Group_1, 2, 10, 10);
Group_1 = add_step(Group_1, 4, 10, pi/2);
Group_1 = add_step(Group_1, 3, 20, -5);

% Descend and re-orientate and waypoint 1
Group_1 = add_step(Group_1, 3, 30, 3);
Group_1 = add_step(Group_1, 4, 30, pi/2);

% Fly to waypoint 2
Group_1 = add_step(Group_1, 1, 35, -10);

% Return to initial waypoint
Group_1 = add_step(Group_1, 1, 55, 10);
Group_1 = add_step(Group_1, 2, 55, -10);
Group_1 = add_step(Group_1, 4, 55, 0.75*pi);
Group_1 = add_step(Group_1, 3, 55, 1);

% Write command to file
save('mainModels/Position_Attitude_Reference.mat', 'Group_1');

% Simulate flight
simout = sim('mainModels/parrotMinidroneHover.slx', tFinal);

% Visualise the resulting flight trajectory
visualise_sim(simout, Group_1, state_names, input_names);

% Extract and save data
save_sim('../data/Waypoint Pathing.mat', simout);


%% Tidy up

% Close project
close(proj)


%% Helper Functions

function ref = new_signal(ref, tFinal)

	% Define reference signal names
	refnames = {'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'};

	% Reset signal
	for idx = 1:6
		ref{idx} = timeseries([0; 0], [0, tFinal], Name=refnames{idx});
	end

	% Add takeoff command
	ref = add_step(ref, 3, 1, -1);

end

function ref = add_step(ref, signal_idx, step_time, step_size)

	% Extract timeseries data
	ts = ref{signal_idx};

	% Get index before step time
	preStep_idx = ts.Time < step_time;

	% Get the signal value before step time
	preStep_value = ts.Data(sum(preStep_idx));
	postStep_value = preStep_value + step_size;

	% Get tFinal from signal
	tFinal = ts.Time(end);

	% Construct new time and data signals
	t_new = [ts.Time(preStep_idx); step_time; step_time; tFinal];
	data_new = [ts.Data(preStep_idx); preStep_value; postStep_value; postStep_value];

	% Overwrite timeseries data with new timeseries
	ref{signal_idx} = timeseries(data_new, t_new, Name=ts.Name);

end

function ref = add_chirp(ref, signal_idx, start_time, duration, min_freq, max_freq, dt, amplitude)
    
	% Extract timeseries data
    ts = ref{signal_idx};
    
    % Get index before start time
    preStart_idx = ts.Time < start_time;
    
    % Get the baseline signal value to oscillate around
    baseline_value = ts.Data(sum(preStart_idx));
    
    % Get tFinal from the original signal
    tFinal = ts.Time(end);
    end_time = start_time + duration;
    
    % Generate high-resolution time vector for the chirp
    t_chirp = (start_time:dt:end_time)';
    
    % Ensure the exact end time is included to prevent truncation
    if t_chirp(end) ~= end_time
        t_chirp = [t_chirp; end_time];
    end
    
    % Calculate relative time for the phase calculation
    t_rel = t_chirp - start_time;
    
    % Calculate the linear frequency sweep rate (Hz per second)
    c = (max_freq - min_freq) / duration;
    
    % Construct the linear chirp data
    % Formula: x(t) = A * sin(2 * pi * (f_0 + (c/2)*t) * t)
    chirp_data = baseline_value + amplitude * sin(2 * pi * (min_freq + 0.5 * c * t_rel) .* t_rel);
    
    % Construct new time and data signals
    % The signal returns exactly to the baseline value after the chirp concludes
    t_new = [ts.Time(preStart_idx); t_chirp; tFinal];
    data_new = [ts.Data(preStart_idx); chirp_data; baseline_value];
    
    % Overwrite timeseries data with new timeseries
    ref{signal_idx} = timeseries(data_new, t_new, Name=ts.Name);

end

function visualise_sim(simout, ref, state_names, input_names)

	figure(WindowState='maximized');
	for idx = 1:12
	
		subplot(16, 1, idx);
		plot(simout.estimatedStates.time, simout.estimatedStates.signals.values(:, idx), LineWidth=1);
		hold on;
		
		if idx < 7
			plot(ref{idx}.Time, ref{idx}.Data, '--');
		end
	
		hold off;
		grid on;
		grid minor;
		ylabel(state_names{idx});
	
	end
	
	for idx = 1:4
		
		subplot(16, 1, idx+12);
		plot(simout.inputs.Time, simout.inputs.Data(:, idx), LineWidth=1);
		grid on;
		grid minor;
		ylabel(input_names{idx});
	
	end
	
	xlabel('Time (s)');

end

function save_sim(savedir, simout)

	t = simout.estimatedStates.time;
	x = simout.estimatedStates.signals.values;
	x(:,4) = simout.estimatedStates.signals.values(:,6);
	x(:,6) = simout.estimatedStates.signals.values(:,4);
	u = simout.inputs.Data;
	ref = [simout.ref.pos_ref.Data, simout.ref.orient_ref.Data(:,end:-1:1)];
	save(savedir, 't', 'x', 'u', 'ref');

end