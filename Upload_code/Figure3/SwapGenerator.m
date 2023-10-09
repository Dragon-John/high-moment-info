function mat=SwapGenerator(n)

state_0 = [1
           0];

state_1 = [0
           1];

state = {state_0, state_1};

if n ==2
    mat = zeros(4);
    for i = [1:2]
        for j = [1:2]
                mat = mat + tensor(state{j}, state{i}) * tensor(state{i}, state{j})';
        end
    end
end


if n == 3
    mat = zeros(8);
    for i = [1:2]
        for j = [1:2]
            for t = [1:2]
                mat = mat + tensor(state{t}, state{i}, state{j}) * tensor(state{i}, state{j}, state{t})';
            end
        end
    end
end

if n==4
    mat = zeros(16);
    for i = [1:2]
        for j = [1:2]
            for t = [1:2]
                for k = [1:2]
                mat = mat + tensor(state{k}, state{i}, state{j}, state{t}) * tensor(state{i}, state{j}, state{t}, state{k})';
                end
            end
        end
    end
end

if n==5
    mat = zeros(32);
    for i = [1:2]
        for j = [1:2]
            for k = [1:2]
                for t = [1:2]
                    for m = [1:2]
                        mat = mat + tensor(state{m}, state{i}, state{j}, state{k}, state{t}) * tensor(state{i}, state{j}, state{k}, state{t}, state{m})';
                    end
                end
            end
        end
    end
end

if n==6
    mat = zeros(64);
    for i = [1:2]
        for j = [1:2]
            for k = [1:2]
                for t = [1:2]
                    for m = [1:2]
                        for n = [1:2]
                        mat = mat + tensor(state{n}, state{i}, state{j}, state{k}, state{t}, state{m}) * tensor(state{i}, state{j}, state{k}, state{t}, state{m}, state{n})';
                        end
                    end
                end
            end
        end
    end
end

% if n=='twoqubit'
%     mat = zeros(16);
%     for i = [1:2]
%         for j = [1:2]
%             for t = [1:2]
%                 for k = [1:2]
%                 mat = mat + tensor(state{i}, state{j}, state{k}, state{t}) * tensor(state{k}, state{t}, state{i}, state{j})';
%                 end
%             end
%         end
%     end
% end
% 
% if n=='threequbit'
%     mat = zeros(64);
%     for i = [1:2]
%         for j = [1:2]
%             for k = [1:2]
%                 for t = [1:2]
%                     for m = [1:2]
%                         for n = [1:2]
%                         mat = mat + tensor(state{t}, state{m}, state{n}, state{i}, state{j}, state{k}) * tensor(state{i}, state{j}, state{k}, state{t}, state{m}, state{n})';
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end