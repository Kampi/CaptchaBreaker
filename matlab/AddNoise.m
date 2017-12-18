clc;
clear;

% Settings for the noise (Variance in dB)
SNR = 100;
Mean = 0.0;

% Project directory
ProjectDir = 'D:\Dropbox\GitHub\Machine-Learning\CaptchaBreaker';

% Create full path
FilePath = strcat(ProjectDir, '\data\test\');

% Read files from directory
Files = dir(FilePath);

% Create new directory
NewPath = strcat(ProjectDir, '\data\noise\', num2str(SNR), '_dB');
mkdir(NewPath)

for File = 1:length(Files)

   Size = Files(File).bytes;
   
   if(Size > 0)   
        FileName = strcat(FilePath, Files(File).name);
        
        % Read image
        ImageBefore = imread(FileName);
        
        % Get the image intensity
        Intensity = double(ImageBefore);
        
        % Adjust intensity between 0 and 1
        Intensity = Intensity - min(Intensity(:));
        Intensity = Intensity / max(Intensity(:));
        
        % Convert variance to noise
        % SNR = 10*log10(Variance_Image / Variance_Noise)
        Variance = var(Intensity(:)) / (10^(SNR / 10));

        % Add noise to the image
        ImageAfter = imnoise(ImageBefore, 'gaussian', Mean, Variance);
        
        % Store the new image
        ImagePath = strcat(NewPath, '\', Files(File).name, '.jpg');
        imwrite(ImageAfter, ImagePath);
   end
end