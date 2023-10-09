%% Please run this file first to get the data, and then, execute the PlotFigure.m to plot

clear;
k=3;
channel_noise = 0:0.05:0.5;

method_1_cost_AD = [];
method_2_cost_AD = [];
method_3_cost_AD = [];

method_1_cost_DE = [];
method_2_cost_DE = [];
method_3_cost_DE = [];

for i = channel_noise
    [cost, JD] = Method_1(k, 'AD', i);
    method_1_cost_AD(end+1) = cost;
    % [cost, JD] = Method_2(k, 'AD', i);
    % method_2_cost_AD(end+1) = cost;
    [cost, JD] = Method_3(k, 'AD', i);
    method_3_cost_AD(end+1) = cost;

    [cost, JD] = Method_1(k, 'DE', i);
    method_1_cost_DE(end+1) = cost;
    % [cost, JD] = Method_2(k, 'DE', i);
    % method_2_cost_DE(end+1) = cost;
    [cost, JD] = Method_3(k, 'DE', i);
    method_3_cost_DE(end+1) = cost;
end
