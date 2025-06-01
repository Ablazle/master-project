clc;
fprintf('\nMAT-to-SET batch converter started at %s\n\n', datestr(now));
addpath(genpath('/home/au682014/biosig4octmat'));
addpath('/home/au682014/eeglab_project/eeglab');
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Mapping table: filename, subject folder, session folder
mapping = {
    '107_2_OnlyPPChans.mat', 'sub-24390', 'ses-01';
    '127_3_OnlyPPChans.mat', 'sub-36821', 'ses-01';
    '142_3_OnlyPPChans.mat', 'sub-18657', 'ses-01';
    '152_2_OnlyPPChans.mat', 'sub-22519', 'ses-01';
    '158_3_OnlyPPChans.mat', 'sub-21502', 'ses-01';
    '229_2_OnlyPPChans.mat', 'sub-23608', 'ses-01';
    '260_2_OnlyPPChans.mat', 'sub-15749', 'ses-01';
    '263_2_OnlyPPChans.mat', 'sub-22073', 'ses-01';
    '278_3_OnlyPPChans.mat', 'sub-35712', 'ses-01';
    '283_1_OnlyPPChans.mat', 'sub-20619', 'ses-01';
    '298_1_OnlyPPChans.mat', 'sub-25813', 'ses-01';
    '345_1_OnlyPPChans.mat', 'sub-29387', 'ses-01';
    '360_3_OnlyPPChans.mat', 'sub-29708', 'ses-01';
    '400_2_OnlyPPChans.mat', 'sub-13167', 'ses-01';
    '422_3_OnlyPPChans.mat', 'sub-17829', 'ses-01';
    '437_1_OnlyPPChans.mat', 'sub-24263', 'ses-01';
    '443_2_OnlyPPChans.mat', 'sub-22987', 'ses-01';
    '451_2_OnlyPPChans.mat', 'sub-23906', 'ses-01';
    '462_1_OnlyPPChans.mat', 'sub-21870', 'ses-01';
    '485_3_OnlyPPChans.mat', 'sub-20634', 'ses-01';
    '538_2_OnlyPPChans.mat', 'sub-25394', 'ses-01';
    '547_3_OnlyPPChans.mat', 'sub-34701', 'ses-01';
    '549_2_OnlyPPChans.mat', 'sub-21388', 'ses-01';
    '582_1_OnlyPPChans.mat', 'sub-20347', 'ses-01';
    '654_2_OnlyPPChans.mat', 'sub-20365', 'ses-01';
    '656_3_OnlyPPChans.mat', 'sub-30985', 'ses-01';
    '659_1_OnlyPPChans.mat', 'sub-21753', 'ses-01';
    '679_3_OnlyPPChans.mat', 'sub-32801', 'ses-01';
    '689_2_OnlyPPChans.mat', 'sub-23842', 'ses-01';
    '726_1_OnlyPPChans.mat', 'sub-29732', 'ses-01';
    '748_2_OnlyPPChans.mat', 'sub-26530', 'ses-01';
    '749_1_OnlyPPChans.mat', 'sub-23152', 'ses-01';
    '775_2_OnlyPPChans.mat', 'sub-18423', 'ses-01';
    '784_2_OnlyPPChans.mat', 'sub-25634', 'ses-01';
    '800_3_OnlyPPChans.mat', 'sub-29341', 'ses-01';
    '815_3_OnlyPPChans.mat', 'sub-29827', 'ses-01';
    '817_2_OnlyPPChans.mat', 'sub-14890', 'ses-01';
    '826_2_OnlyPPChans.mat', 'sub-25219', 'ses-01';
    '841_2_OnlyPPChans.mat', 'sub-26374', 'ses-01';
    '843_2_OnlyPPChans.mat', 'sub-17098', 'ses-01';
    '849_3_OnlyPPChans.mat', 'sub-31976', 'ses-01';
    '852_1_OnlyPPChans.mat', 'sub-23650', 'ses-01';
    '869_2_OnlyPPChans.mat', 'sub-14360', 'ses-01';
    '876_1_OnlyPPChans.mat', 'sub-25316', 'ses-01';
    '913_2_OnlyPPChans.mat', 'sub-26578', 'ses-01';
    '913_3_OnlyPPChans.mat', 'sub-37915', 'ses-01';
    '916_3_OnlyPPChans.mat', 'sub-19742', 'ses-01';
    '934_3_OnlyPPChans.mat', 'sub-39201', 'ses-01';
    '942_1_OnlyPPChans.mat', 'sub-22176', 'ses-01';
    '957_3_OnlyPPChans.mat', 'sub-21235', 'ses-01';
    '958_3_OnlyPPChans.mat', 'sub-21847', 'ses-01';
    '959_3_OnlyPPChans.mat', 'sub-37790', 'ses-01';
    '965_3_OnlyPPChans.mat', 'sub-22964', 'ses-01';
    '996_2_OnlyPPChans.mat', 'sub-25609', 'ses-01';
    '997_1_OnlyPPChans.mat', 'sub-22873', 'ses-01';
};

inputDir = '/home/au682014/eeglab_project/RData/';
outputBaseDir = '/home/au682014/eeglab_project/set_files/';

for i = 1:size(mapping, 1)
    filename     = mapping{i,1};
    subjectFold  = mapping{i,2};
    sessionFold  = mapping{i,3};
    inPath       = fullfile(inputDir, filename);
    outFolder    = fullfile(outputBaseDir, subjectFold, sessionFold);

    fprintf('\n---------------------------\n');
    fprintf('Processing file: %s\n', inPath);
    try
        S = load(inPath);
        if ~isfield(S, 'EEG')
            fprintf('  [Skipping] No ''EEG'' struct found in %s\n', inPath);
            continue;
        end
        EEG = S.EEG;
        EEG = eeg_checkset(EEG);

        requiredFields = {'data','nbchan','srate','chanlocs'};
        if ~all(isfield(EEG, requiredFields))
            fprintf('  [Skipping] EEG struct missing required fields.\n');
            continue;
        end
        if size(EEG.data,1) < 12
            fprintf('  [Skipping] Less than 12 channels present.\n');
            continue;
        end

        if ~exist(outFolder, 'dir')
            mkdir(outFolder);
            fprintf('  Created output folder: %s\n', outFolder);
        end
        subjectID = regexp(subjectFold, '\d+', 'match', 'once');
        sessionID = regexp(sessionFold, '\d+', 'match', 'once');
        outFile   = sprintf('sub-%s_ses-%s_task-sleep_eeg.set', subjectID, sessionID);
        outPath   = fullfile(outFolder, outFile);

        % Keep only first 12 channels
        fprintf('  Keeping first 12 channels...\n');
        EEG = pop_select(EEG, 'channel', 1:12);
        EEG = eeg_checkset(EEG);

        % Replace NaNs and Inf with zeros
        fprintf('  Replacing NaNs/Infs with zeros if present...\n');
        EEG.data(isnan(EEG.data)) = 0;
        EEG.data(isinf(EEG.data)) = 0;

        % Ensure data is double precision
        if ~isa(EEG.data, 'double')
            fprintf('    EEG.data not double, converting to double.\n');
            EEG.data = double(EEG.data);
        end

        % Channel label diagnostics and fix
        fprintf('  Channel label pre-export diagnostics:\n');
        fprintf('    EEG.nbchan = %d, size(EEG.data,1) = %d, length(EEG.chanlocs) = %d\n', ...
            EEG.nbchan, size(EEG.data,1), length(EEG.chanlocs));
        for k = 1:EEG.nbchan
            if isfield(EEG.chanlocs(k), 'labels') && ~isempty(EEG.chanlocs(k).labels)
                EEG.chanlocs(k).labels = char(EEG.chanlocs(k).labels);
            else
                EEG.chanlocs(k).labels = ['Ch' num2str(k)];
                fprintf('    Warning: Chan %d had missing/empty label, set to ''Ch%d''\n', k, k);
            end
            fprintf('    Chan %02d: %s\n', k, EEG.chanlocs(k).labels);
        end

        labels = {EEG.chanlocs.labels};
        if length(unique(labels)) ~= EEG.nbchan
            fprintf('    Warning: Channel labels are not unique. Forcing uniqueness.\n');
            for k = 1:EEG.nbchan
                EEG.chanlocs(k).labels = sprintf('Ch%02d', k);
            end
        end

        if isfield(EEG, 'urchanlocs')
            EEG = rmfield(EEG, 'urchanlocs');
        end

        fprintf('    Final check: nbchan=%d, data=%dx%d, chanlocs=%d\n', EEG.nbchan, size(EEG.data,1), size(EEG.data,2), length(EEG.chanlocs));

        % RESAMPLE TO 128 HZ
        try
            EEG = pop_resample(EEG, 128);
            EEG = eeg_checkset(EEG);
            fprintf('    Successfully resampled to 128 Hz.\n');
        catch ME
            fprintf('  [Error] Could not resample: %s\n', ME.message);
            continue;
        end

        % Save as SET (overwriting if present)
        fprintf('  Saving as SET: %s\n', outPath);
        try
            EEG = pop_saveset(EEG, 'filename', outFile, 'filepath', outFolder, 'savemode', 'onefile');
            fprintf('    Successfully wrote SET file.\n');
        catch ME
            fprintf('  [Error] Could not save SET file: %s\n', ME.message);
            continue;
        end

        fprintf('  DONE: Converted, saved, and resampled %s\n', outPath);

    catch ME
        fprintf('  [Error] %s\n', ME.message);
        continue;
    end
end

fprintf('\nMAT-to-SET conversion process completed at %s\n', datestr(now));
