%% Plot the figure. Before Executing this code, please run MainExperiments first
color_1 = [115/255, 209/255, 233/255]; % Define the color
color_2 = [46/255, 47/255, 103/255]; % Define the color

grid on
hold on
plot(channel_noise, method_1_cost_AD,'--x','LineWidth',2,'Color',color_1);
plot(channel_noise, method_1_cost_DE,'--x','LineWidth',2,'Color',color_2);

plot(channel_noise, method_3_cost_AD,'-x','LineWidth',2,'Color',color_1);
plot(channel_noise, method_3_cost_DE,'-x','LineWidth',2,'Color',color_2);

leg = legend('Channel inverse (AD)','Channel inverse (DE)', 'New protocol (AD)', 'New protocol (DE)');
set(leg,'Interpreter','latex','FontSize',16,'Location','northwest');
xlabel('Noise level','FontSize', 20, 'FontName','Times New Roman')
ylabel('Overhead','FontSize',20, 'FontName','Times New Roman')
