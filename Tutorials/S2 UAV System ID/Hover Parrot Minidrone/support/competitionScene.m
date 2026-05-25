% @brief Sets up the competition scene with floor, pillars, walls, lines, and shapes.
% @param Actor The main actor in the scene.
% @param World The simulation world where actors are added.
function competitionScene(Actor, World)

	% @brief Adds a grey floor to the scene.
    floor = sim3d.Actor('ActorName', 'floor');
    createShape(floor, 'box', [5 5 0.1]); % 5x5 meters floor with 0.1 meters thickness
    floor.Color = [0.37 0.5 0.33]; % Grey color
    floor.Metallic = 0.1;
    floor.Specular = 0.1;
    floor.Flat = 1;
    floor.Friction = 0.7;
    floor.Restitution = 0.3;
    floor.Translation = [2 2 -0.05];
    floor.Texture = fullfile(slproject.getCurrentProject().RootFolder,"support","texture","gravel.jpg");
    World.add(floor, Actor);

	% @brief Adds four pillars at the corners of the arena.
    pillar_positions = [0 0; 4 0; 0 4; 4 4];
    for i = 1:4
        pillar = sim3d.Actor('ActorName', ['pillar' num2str(i)]);
        createShape(pillar, 'cylinder', [0.3 0.3 3]); % Height of 3 meters
        pillar.Translation = [pillar_positions(i, :) 1.5];
        pillar.Color = [0 0 0];
        pillar.Collisions = 0;
        propagate(pillar, 'OverlapEventEnabled', true);
        World.add(pillar, Actor);
    end

	% @brief Creates a solid wall in the scene.
	% @param world The simulation world.
	% @param actor The actor to which the wall is added.
	% @param baseTranslation The translation of the wall.
	% @param dimensions The dimensions of the wall.
	% @param rotation The rotation of the wall.
    function createSolidWall(world, actor, baseTranslation, dimensions, rotation)
        wall = sim3d.Actor('ActorName', 'wall');
        createShape(wall, 'box', dimensions);
        wall.Color = [1 1 1]; % White color
        wall.Transparency = 0.8; 
        wall.Metallic = 0;
        wall.Shininess = 0;
        wall.Collisions = 0;
        wall.Translation = baseTranslation;
        wall.Rotation = rotation;
        propagate(wall, 'OverlapEventEnabled', true);
        world.add(wall, actor);
    end

	% @brief Defines wall parameters and creates the solid walls around the pillars.
    wall_thickness = 0.025; % Thickness of the wall
    wall_height = 3; % Height of the wall

    createSolidWall(World, Actor, [2 4 1.5], [4 wall_thickness wall_height], [0 0 0]);   
    createSolidWall(World, Actor, [2 0 1.5], [4 wall_thickness wall_height], [0 0 0]);  
    createSolidWall(World, Actor, [0 2 1.5], [wall_thickness 4 wall_height], [0 0 0]);  
    createSolidWall(World, Actor, [4 2 1.5], [wall_thickness 4 wall_height], [0 0 0]);  

    initData = struct();
    initFilePath = fullfile(slproject.getCurrentProject().RootFolder, "initData.mat");

	lineWidth = 0.1;

	% @brief Loads or creates initialization data.
    if isfile(initFilePath)
        initData = load(initFilePath);
    else
        initData.Lines = [
            struct('StartX', 0.5, 'StartY', 1, 'EndX', 2.45, 'EndY', 1); % Horizontal part of L
            struct('StartX', 2.45, 'StartY', 1, 'EndX', 2.45, 'EndY', 3) % Vertical part of L
        ];

        initData.Circle = struct('X', 2.45, 'Y', 3.2, 'Z', 0.001, 'Diameter', 0.2); % Circle
    end

	% @brief Creates lines based on initialization data.
    for i = 1:length(initData.Lines)
    	startX = initData.Lines(i).StartX;
    	startY = initData.Lines(i).StartY;
    	endX = initData.Lines(i).EndX;
    	endY = initData.Lines(i).EndY;
	
    	% Calculate the angle of the line
    	angle = atan2(endY - startY, endX - startX);
	
		lineLength = sqrt((endX - startX)^2 + (endY - startY)^2);

    	% Create the line actor
    	line_actor = sim3d.Actor('ActorName', ['line_' num2str(i)]);
    	createShape(line_actor, 'box', [lineLength lineWidth 0.001]); % Line thickness
	
    	% Set the color
    	if isfile(initFilePath)
        	line_actor.Color = initData.Color;
    	else
        	line_actor.Color = [1 0 0];
    	end
	
		% Set the position to the midpoint of the line
    	midX = (startX + endX) / 2;
    	midY = (startY + endY) / 2;
    	line_actor.Translation = [midX midY 0.001];
	
    	% Set the rotation based on the angle
    	line_actor.Rotation = [0 0 angle];
        if isfile(initFilePath)
            line_actor.Color = initData.Color;
        else
            line_actor.Color = [1 0 0];
        end
        % Add the line actor to the world
        World.add(line_actor, Actor);
    end
    
	% Draw disks at each joint (except endpoints)
	for i = 2:length(initData.Lines)
    	jointX = initData.Lines(i).StartX;
    	jointY = initData.Lines(i).StartY;
	
    	corner_actor = sim3d.Actor('ActorName', ['corner_' num2str(i)]);
    	createShape(corner_actor, 'cylinder', [lineWidth lineWidth 0.001]);

		if isfile(initFilePath)
			corner_actor.Color = initData.Color;
		else
			corner_actor.Color = [1 0 0];
		end

    	corner_actor.Translation = [jointX, jointY, 0.001];
    	corner_actor.Rotation = [0 0 0];
    	World.add(corner_actor, Actor);
	end

	% @brief Creates the red circle at the end of the L-shape.
    red_circle = sim3d.Actor('ActorName', 'red_circle');
    createShape(red_circle, 'cylinder', [initData.Circle.Diameter initData.Circle.Diameter 0.001]); 
    if isfile(initFilePath)
        red_circle.Color = initData.Color;
    else
        red_circle.Color = [1 0 0];
    end

    red_circle.Translation = [initData.Circle.X initData.Circle.Y initData.Circle.Z]; 
    World.add(red_circle, Actor);
    
	% @brief Creates the yellow square at the beginning of the L-shape.
    yellow_square = sim3d.Actor('ActorName', 'yellow_square');
    createShape(yellow_square, 'box', [0.2 0.2 0.01]); % 20cm x 20cm, paper thickness
    yellow_square.Color = [1 1 0]; % Yellow color
    yellow_square.Shininess = 0.2;
    yellow_square.Metallic = 0;
    yellow_square.Specular = 0.5;
    yellow_square.Friction = 0.7;
    yellow_square.Restitution = 0.3;
    yellow_square.Translation = [0.5 2 0.02];
    World.add(yellow_square, Actor);
    
	% @brief Creates the blue square at the beginning of the L-shape.
    blue_square = sim3d.Actor('ActorName', 'blue_square');
    createShape(blue_square, 'box', [0.2 0.2 0.01]); % 20cm x 20cm, paper thickness
    blue_square.Color = [0 0 1]; % Blue color
    blue_square.Shininess = 0.2;
    blue_square.Metallic = 0;
    blue_square.Specular = 0.5;
    blue_square.Friction = 0.7;
    blue_square.Restitution = 0.3;
    blue_square.Translation = [0.5 2.5 0.02];
    World.add(blue_square, Actor);
end