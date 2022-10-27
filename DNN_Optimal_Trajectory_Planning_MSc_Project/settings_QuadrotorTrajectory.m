function options = settings_QuadrotorTrajectory(varargin)
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
% This function provides various settings to the solver.
%
% Syntax:  options = settings_QuadrotorTrajectory(varargin)
%          When giving one input with varargin, e.g. with settings(20), will use h-method of your choice with N=20 nodes
%          When giving two inputs with varargin, hp-LGR method will be used with two possibilities
%           - Scalar numbers can be used for equally spaced intervals with the same polynoimial orders. For example, settings_hp(5,4) means using 5 LGR intervals each of polynomial degree of 4. 
%           - Alternatively, can supply two arrays in the argument with customized meshing. For example, settings_hp([-1 0.3 0.4 1],[4 5 3]) will have 3 segments on the normalized interval [-1 0.3], [0.3 0.4] and [0.4 1], with polynomial order of 4, 5, and 3 respectively.
%      
% Output:
%    options - Structure containing the settings
%


%% Transcription Method

% Direct_collocation
options.transcription = 'direct_collocation';

% Integrated residual minimization : alternating method
options.min_res_mode = 'alternating';

% Priorities for the solution property : lower integrated residual error
options.min_res_priority = 'low_res_error';

% Error criteria : local absolute error                        
options.errortype='local_abs';


%% Discretization Method

% Discretization method : Hermite-Simpson method 
options.discretization = 'hermite';

% Result Representation: direct interpolation in correspondence with the transcription method  
options.resultRep = 'default';

%% Derivative generation

% Derivative computation method : finite differences  ('numeric')
options.derivatives = 'numeric';
options.adigatorPath = '../../adigator';

% Perturbation sizes for numerical differentiation
options.perturbation.H = []; 
options.perturbation.J = []; 

%% NLP solver

% NLP solver : IPOPT                           
options.NLPsolver = 'ipopt';

% IPOPT settings 
options.ipopt.tol = 1e-9;                        % Convergence tolerance (relative).
options.ipopt.print_level = 5;                   % Print level. 
options.ipopt.max_iter = 5000;                   % Maximum number of iterations. 
 
options.ipopt.mu_strategy = 'adaptive';         % Adaptive update strategy.    

options.ipopt.hessian_approximation = 'exact';   %  Use second derivatives provided by ICLOCS.                                                 

options.ipopt.limited_memory_max_history = 6;   % Maximum size of the history for the limited quasi-Newton Hessian approximation. 

options.ipopt.limited_memory_max_skipping = 1;  % Threshold for successive iterations where update is skipped for the quasi-Newton approximation.


%% Meshing Strategy

% Type of meshing : with local refinement
options.meshstrategy = 'mesh_refinement';

% Mesh Refinement Method : Automatic refinement
options.MeshRefinement = 'Auto';

% Mesh Refinement Preferences : efficient
options.MRstrategy = 'efficient';

% Maximum number of mesh refinement iterations
options.maxMRiter = 50;

% Discountious Input
options.disContInputs = 0;

% Minimum and maximum time interval
options.mintimeinterval = 0.001; 
options.maxtimeinterval = inf; 

% Distribution of integration steps : equispaced steps. 
options.tau = 0;


%% Other Settings

% Cold/Warm/Hot Start 
options.start = 'Cold';

% Automatic scaling 
options.scaling = 0;

% Reorder of LGR Method
options.reorderLGR = 0;

% Early termination of residual minimization if tolerance is met
options.resminEarlyStop = 0;

% External Constraint Handling
options.ECH.enabled = 0; 

% A conservative buffer zone 
options.ECH.buffer_pct = 0.1; 

% Regularization Strategy
options.regstrategy = 'off';

% LEAVE THIS PART UNCHANGED AND USE FUNCTION SYNTAX (AS DESCRIBED ON THE TOP) TO DEFINE THE ITEGRATION NODES
if nargin==2
    if strcmp(varargin{2},'h')
        options.nodes=varargin{1}; 
        options.discretization='hermite';
    else
        if length(varargin{1})==1
            options.nsegment=varargin{1}; 
            options.pdegree=varargin{2}; 
        else
            options.tau_segment=varargin{1};
            options.npsegment=varargin{2};
        end
        options.discretization='hpLGR';
    end
else
    options.nodes=varargin{1}; 
end


%% Output settings

% Display computation time
options.print.time = 1;

% Display relative local discretization error 
options.print.relative_local_error = 1;

% Display cost (objective) values
options.print.cost = 1;


% Plot figures : plot all figures
options.plot = 1;