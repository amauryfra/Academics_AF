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
% This is the main solver script.



clear all;
close all;
format compact;

[problem,guess] = QuadrotorTrajectory;          % Problem definition
options = problem.settings(20);                 % Get options and solver settings 
[solution,MRHistory] = solveMyProblem(problem,guess,options);
[ tv, xv, uv ] = simulateSolution( problem, solution, 'ode113', 0.001 );

%% figure

xx = linspace(solution.T(1,1),solution.T(end,1),100);

figure
plot(speval(solution,'X',1,xx), speval(solution,'X',3,xx), 'r-' )
xlabel('x [m]')
ylabel('z [m]')
grid on

figure
plot(xx,speval(solution,'U',1,xx),'b-' )
hold on
plot(xx,speval(solution,'U',2,xx),'r-' )
plot(tv,uv(:,1),'k-.')
plot(tv,uv(:,2),'k-.')
plot([solution.T(1,1); solution.tf],[problem.inputs.ul(1), problem.inputs.ul(1)],'r-' )
plot([solution.T(1,1); solution.tf],[problem.inputs.uu(1), problem.inputs.uu(1)],'r-' )
xlim([0 solution.tf])
xlabel('Time [s]')
grid on
ylabel('Control Input')
legend('uT','uR')
