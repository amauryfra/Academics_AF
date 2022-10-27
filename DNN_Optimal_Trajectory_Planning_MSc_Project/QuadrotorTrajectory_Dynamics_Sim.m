function [dx] = QuadrotorTrajectory_Dynamics_Sim(x,u,p,t,data)
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
% Syntax:  
%          [dx] = QuadrotorTrajectory_Dynamics_Sim(x,u,p,t,vdat)	(Dynamics Only)
% 
% Inputs:
%    x  - state vector
%    u  - input
%    p  - parameter
%    t  - time
%    vdat - structured variable containing the values of additional data used inside
%          the function      
% Output:
%    dx - time derivative of x
%

% State variables
x1 = x(:,1); % x
x2 = x(:,2); % xDot
x3 = x(:,3); % z
x4 = x(:,4); % zDot
x5 = x(:,5); % theta

% Controls
uT = u(:,1);
uR = u(:,2);

% Dynamics

dx(:,1) = x2;

dx(:,2) = uT .* sin(x5);

dx(:,3) = x4;

dx(:,4) = uT .* cos(x5) - 9.81;

dx(:,5) = uR;

