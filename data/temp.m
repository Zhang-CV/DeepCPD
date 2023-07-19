% fid = fopen('eval_data.txt', 'a');
class = 'GPM';
% for num = 1 : 4*128
%     fprintf(fid, string(class) + '_' + string(num - 1) + '.ply\n');
% end
fid = fopen('train_data.txt', 'a');
for num = 1 : 8*1024
    fprintf(fid, string(class) + '_' + string(num - 1) + '.ply\n');
end
% fid = fopen('test_data.txt', 'a');
% for num = 1 : 4*1024
%     fprintf(fid, string(class) + '_' + string(num - 1) + '.ply\n');
% end