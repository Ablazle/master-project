clc;
clear;

% Define mapping (filename to subject/session folder)
mapping = {
    '775_2_OnlyPPChans.mat', 'sub-775',   'ses-01'
    '817_2_OnlyPPChans.mat', 'sub-817',   'ses-01'
    '869_2_OnlyPPChans.mat', 'sub-869',   'ses-01'
    '843_2_OnlyPPChans.mat', 'sub-843',   'ses-01'
    '400_2_OnlyPPChans.mat', 'sub-400',   'ses-01'
    '260_2_OnlyPPChans.mat', 'sub-260',   'ses-01'
    '852_1_OnlyPPChans.mat', 'sub-852',   'ses-01'
    '742_1_OnlyPPChans.mat', 'sub-742',   'ses-01'
    '942_1_OnlyPPChans.mat', 'sub-856',   'ses-02'
    '360_3_OnlyPPChans.mat', 'sub-360',   'ses-01'
    '952_1_OnlyPPChans.mat', 'sub-360',   'ses-03'
    '876_1_OnlyPPChans.mat', 'sub-876',   'ses-01'
    '749_1_OnlyPPChans.mat', 'sub-749',   'ses-01'
    '283_1_OnlyPPChans.mat', 'sub-749',   'ses-02'
    '437_1_OnlyPPChans.mat', 'sub-749',   'ses-03'
    '462_1_OnlyPPChans.mat', 'sub-462',   'ses-01'
    '726_1_OnlyPPChans.mat', 'sub-462',   'ses-02'
    '997_1_OnlyPPChans.mat', 'sub-462',   'ses-03'
    '263_2_OnlyPPChans.mat', 'sub-263',   'ses-01'
    '654_2_OnlyPPChans.mat', 'sub-263',   'ses-03'
    '689_2_OnlyPPChans.mat', 'sub-689',   'ses-01'
    '748_2_OnlyPPChans.mat', 'sub-689',   'ses-02'
    '451_2_OnlyPPChans.mat', 'sub-451',   'ses-01'
    '841_2_OnlyPPChans.mat', 'sub-451',   'ses-02'
    '229_2_OnlyPPChans.mat', 'sub-451',   'ses-03'
    '913_2_OnlyPPChans.mat', 'sub-913',   'ses-01'
    '152_2_OnlyPPChans.mat', 'sub-913',   'ses-02'
    '538_2_OnlyPPChans.mat', 'sub-538',   'ses-01'
    '784_2_OnlyPPChans.mat', 'sub-538',   'ses-03'
    '443_2_OnlyPPChans.mat', 'sub-443',   'ses-01'
    '962_2_OnlyPPChans.mat', 'sub-962',   'ses-01'
    '958_3_OnlyPPChans.mat', 'sub-958',   'ses-01'
    '965_3_OnlyPPChans.mat', 'sub-958',   'ses-02'
    '158_3_OnlyPPChans.mat', 'sub-958',   'ses-03'
    '957_3_OnlyPPChans.mat', 'sub-971',   'ses-02'
    '800_3_OnlyPPChans.mat', 'sub-800',   'ses-01'
    '549_2_OnlyPPChans.mat', 'sub-549',   'ses-01'
    '849_3_OnlyPPChans.mat', 'sub-849',   'ses-01'
    '934_3_OnlyPPChans.mat', 'sub-849',   'ses-02'
    '679_3_OnlyPPChans.mat', 'sub-849',   'ses-03'
    '792_3_OnlyPPChans.mat', 'sub-3422',  'ses-03'
};

inputDir = '/home/au682014/eeglab_project/RData/';
outputDir = '/home/au682014/eeglab_project/matinfo/';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for i = 1:size(mapping, 1)
    filename    = mapping{i,1};
    subjectFold = mapping{i,2};
    sessionFold = mapping{i,3};
    matPath     = fullfile(inputDir, filename);
    outFile     = fullfile(outputDir, sprintf('%s_%s_matinfo.txt', subjectFold, sessionFold));
    
    fprintf('Inspecting %s ...\n', matPath);
    
    try
        S = load(matPath);
        if ~isfield(S, 'EEG')
            fprintf('  [Skipping] No EEG struct found in %s\n', matPath);
            continue;
        end
        EEG = S.EEG;
        
        fid = fopen(outFile, 'w');
        fprintf(fid, 'MAT file: %s\n', matPath);
        fprintf(fid, 'Subject: %s, Session: %s\n\n', subjectFold, sessionFold);
        print_struct_field(fid, EEG, 0);
        fclose(fid);
    catch ME
        fprintf('  [ERROR] Could not process file: %s\n  %s\n', matPath, ME.message);
        continue;
    end
end

function print_struct_field(fid, s, depth)
    if depth > 2
        fprintf(fid, '%s<...Nested too deep>\n', repmat('  ', 1, depth));
        return;
    end
    fields = fieldnames(s);
    for j = 1:length(fields)
        fname = fields{j};
        val = s.(fname);
        indent = repmat('  ', 1, depth);
        try
            if isnumeric(val) || islogical(val)
                sz = size(val);
                if numel(val) > 10
                    fprintf(fid, '%s%s: [%s %s], class=%s\n', indent, fname, num2str(sz(1)), num2str(sz(2)), class(val));
                else
                    fprintf(fid, '%s%s: %s, class=%s\n', indent, fname, mat2str(val), class(val));
                end
            elseif ischar(val)
                fprintf(fid, '%s%s: %s\n', indent, fname, val);
            elseif iscell(val)
                fprintf(fid, '%s%s: {cell array, size=[%s %s]}\n', indent, fname, num2str(size(val,1)), num2str(size(val,2)));
                for k = 1:numel(val)
                    elem = val{k};
                    if ischar(elem)
                        fprintf(fid, '%s  cell{%d}: %s\n', indent, k, elem);
                    elseif isnumeric(elem)
                        sz = size(elem);
                        if numel(elem) > 10
                            fprintf(fid, '%s  cell{%d}: [%s %s] %s\n', indent, k, num2str(sz(1)), num2str(sz(2)), class(elem));
                        else
                            fprintf(fid, '%s  cell{%d}: %s %s\n', indent, k, mat2str(elem), class(elem));
                        end
                    elseif isstruct(elem)
                        fprintf(fid, '%s  cell{%d}: struct\n', indent, k);
                        print_struct_field(fid, elem, depth+2);
                    end
                end
            elseif isstruct(val)
                fprintf(fid, '%s%s: [struct, fields: %s]\n', indent, fname, strjoin(fieldnames(val)', ', '));
                print_struct_field(fid, val, depth+1);
            else
                fprintf(fid, '%s%s: [unhandled class: %s]\n', indent, fname, class(val));
            end
        catch ME2
            fprintf(fid, '%s%s: [ERROR: %s]\n', indent, fname, ME2.message);
        end
    end
end
