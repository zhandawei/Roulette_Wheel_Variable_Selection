clearvars;clc;close all;
% objective function
fun_name = 'Ellipsoid';
% number of variables
num_vari = 100;
% lower and upper bounds
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% number of initial design points
num_initial = 200;
% number of maximum evaluations
max_evaluation = 1000;
% dimension of subspace
sub_vari = 5;
% initial design
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound)+lower_bound;
sample_y = feval(fun_name,sample_x);
evaluation =  size(sample_x,1);
iteration = 1;
[fmin,ind]= min(sample_y);
best_x = sample_x(ind,:);
fprintf('RW-BO on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
while evaluation < max_evaluation
    % train GP models
    GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);
    EI_record = zeros(1,num_vari);
    x_min = zeros(1,num_vari);
    % compute the maximal expected coordinate improvement value for each dimension
    for ii = 1:num_vari
        [x_min(ii),EI_record(ii)]= Optimizer_GA(@(x)-Infill_ECI(x,GP_model,fmin,best_x,ii),1,lower_bound(ii),upper_bound(ii),10,20);
    end
    EI_record = -EI_record;
    % calculate the probability of each dimension
    p = EI_record/sum(EI_record);
    % select the subspace using the roulette wheel selection strategy
    select_dim = randsrc(1,sub_vari,[1:num_vari;p]);
    % optimize the acquisition function in the subspace
    [optimal_x,max_EI]= Optimizer_GA(@(x)-Infill_ECI(x,GP_model,fmin,best_x,select_dim),sub_vari,lower_bound(select_dim),upper_bound(select_dim),20,40);
    % get a new point
    infill_x  = best_x;
    infill_x(:,select_dim) = optimal_x;
    % evaluate the fitness of the acquisition point
    infill_y = feval(fun_name,infill_x);
    iteration = iteration + 1;
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    [fmin,ind]= min(sample_y);
    best_x = sample_x(ind,:);
    evaluation = evaluation + size(infill_x,1);
    fprintf('RW-BO on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
end

