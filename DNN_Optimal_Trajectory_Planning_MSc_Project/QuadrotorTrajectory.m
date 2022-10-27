function [problem,guess] = QuadrotorTrajectory
% Imperial College London
% MSc Applied Mathematics
% This code has been written as part of the MSc project 'Deep Neural Networks 
% for Real-time Trajectory Planning'
% Author : Amaury FRANCOU - CID: 01258326
% Supervisor : Dr Dante KALISE
%
% This code uses the ICLOCS2 optimization based control software in Matlab/Simulink
% (http://www.ee.ic.ac.uk/ICLOCS/default.htm).
%
% It has been inspired by the Two-link robot arm example problem 
% found on the ICLOCS2 website
% (http://www.ee.ic.ac.uk/ICLOCS/ExampleRobotArm.html) and written by
% Yuanbo Nie, Omar Faqir, and Eric Kerrigan. 
%
% Syntax:  [problem,guess] = QuadrotorTrajectory
%
% Outputs:
%    problem - Structure with information on the optimal control problem
%    guess   - Guess for state, control and multipliers.
%

% Defining q0 and qf
q0 = [ 0 0 0 0 0 ];
qf = [ 0 0 1 0 0 ];

% Plant model name, used for Adigator
InternalDynamics = @QuadrotorTrajectory_Dynamics_Internal;
SimDynamics = @QuadrotorTrajectory_Dynamics_Sim;

% Settings file
problem.settings = @settings_QuadrotorTrajectory;

% Initial time t0.
problem.time.t0_min = 0;
problem.time.t0_max = 0;
guess.t0 = 0;

% Final time.
problem.time.tf_min=0.01;     
problem.time.tf_max=inf; 
guess.tf = 2; % Guessing final time

% Initial conditions for system q0.
problem.states.x0 = q0;

% Initial conditions for system.
problem.states.x0l = q0; 
problem.states.x0u = q0; 

% State bounds. 
problem.states.xl = [-inf -inf -inf -inf -inf];
problem.states.xu = [inf inf inf inf inf];

% State error bounds
problem.states.xErrorTol_local = [0.0001 0.0001 0.0001 0.0001 0.0001];
problem.states.xErrorTol_integral = [0.0001 0.0001 0.0001 0.0001 0.0001];
% State constraint error bounds
problem.states.xConstraintTol = [0.0001 0.0001 0.0001 0.0001 0.0001];

% Terminal state bounds qf.
problem.states.xfl = qf; 
problem.states.xfu = qf; 

% Guess the state trajectories with [x0 xf]
guess.states(:,1)=[q0(1) qf(1)]; 
guess.states(:,2)=[q0(2)  qf(2)]; 
guess.states(:,3)=[q0(3)  qf(3)]; 
guess.states(:,4)=[q0(4) qf(4)]; 
guess.states(:,5)=[q0(5) qf(5)];       

% Number of control actions N 
problem.inputs.N = 0;

% Input bounds
uTmax = 14;
uTmin = 0.8;
uRmax = 4.6;

problem.inputs.ul = [uTmin -uRmax];
problem.inputs.uu = [uTmax uRmax];

problem.inputs.u0l = [uTmin 0]; % uR(0) = 0
problem.inputs.u0u = [uTmax 0];

% Input constraint error bounds
problem.inputs.uConstraintTol = [0.0001 0.0001];

% Guess the input sequences with [u0 uf]
guess.inputs(:,1) = [uTmax uTmax];
guess.inputs(:,2) = [0 0];

% Not required
problem.parameters.pl=[];
problem.parameters.pu=[];
guess.parameters=[];
problem.setpoints.states=[];
problem.setpoints.inputs=[];
problem.constraints.ng_eq=0;
problem.constraints.gTol_eq=[];
problem.constraints.gl=[];
problem.constraints.gu=[];
problem.constraints.gTol_neq=[];
problem.constraints.bl=[];
problem.constraints.bu=[];
problem.constraints.bTol=[];

% Problem parameters used in the functions
% Get function handles and return to Main.m
problem.data.InternalDynamics = InternalDynamics;
problem.data.functionfg  =@fg;
problem.data.plantmodel = func2str(InternalDynamics);
problem.functions = {@L,@E,@f,@g,@avrc,@b};
problem.sim.functions = SimDynamics;
problem.sim.inputX = [];
problem.sim.inputU = 1:length(problem.inputs.ul);
problem.functions_unscaled = {@L_unscaled,@E_unscaled,@f_unscaled,@g_unscaled,@avrc,@b_unscaled};
problem.data.functions_unscaled = problem.functions_unscaled;
problem.data.ng_eq = problem.constraints.ng_eq;
problem.constraintErrorTol = ...
    [problem.constraints.gTol_eq,problem.constraints.gTol_neq,...
    problem.constraints.gTol_eq,problem.constraints.gTol_neq,problem.states.xConstraintTol,...
    problem.states.xConstraintTol,problem.inputs.uConstraintTol,problem.inputs.uConstraintTol];


function stageCost = L_unscaled(x,xr,u,ur,p,t,vdat)

% Returns the running cost.
% 
% Syntax:  stageCost = L(x,xr,u,ur,p,t,data)
%
% Inputs:
%    x  - state vector
%    xr - state reference
%    u  - input
%    ur - input reference
%    p  - parameter
%    t  - time
%    data- structured variable containing the values of additional data used inside
%          the function
%
% Output:
%    stageCost - Scalar or vectorized stage cost
%

stageCost =(u(:,1).*u(:,1)+u(:,2).*u(:,2));


function boundaryCost=E_unscaled(x0,xf,u0,uf,p,t0,tf,vdat) 

% Returns the boundary value cost.
%
% Syntax:  boundaryCost = E(x0,xf,u0,uf,p,tf,data)
%
% Inputs:
%    x0  - state at t=0
%    xf  - state at t=tf
%    u0  - input at t=0
%    uf  - input at t=tf
%    p   - parameter
%    tf  - final time
%    data- structured variable containing the values of additional data used inside
%          the function
%
% Output:
%    boundaryCost - Scalar boundary cost
%

boundaryCost = tf;


function bc = b_unscaled(x0,xf,u0,uf,p,t0,tf,vdat,varargin)

% Not useful here. - Returns a column vector containing the evaluation of the
% boundary constraints. 
%
% Syntax:  bc=b(x0,xf,u0,uf,p,tf,data)
%
% Inputs:
%    x0  - state at t=0
%    xf  - state at t=tf
%    u0  - input at t=0
%    uf  - input at t=tf
%    p   - parameter
%    tf  - final time
%    data- structured variable containing the values of additional data used inside
%          the function
%
%          
% Output:
%    bc - column vector containing the evaluation of the boundary function 
%

bc = [];
