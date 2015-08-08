function onyx_app (dataset_prefix, varargin)
    % ONYX_APP (dataset_prefix, ...)
    %
    % Matlab port of the demo application.
    %
    % The training/testing dataset is assumed to be provided in libsvm
    % format, in files named:
    % ${dataset_prefix}-train.data, ${dataset_prefix}-train.labels,
    % ${dataset_prefix}-test.data, and ${dataset_prefix}-test.labels
    %
    % Input:
    %  - dataset_prefix: dataset name prefix
    %  - additional parameters are specified via key/value pairs:
    %     - load_classifier: file to restore classifier from before
    %       training and testing (default: '')
    %     - save_classifier: file to export classifier to after training is
    %       complete (default: '')
    %     - enable_training: enable training (default: true)
    %     - enable_testing: enable testing (default: true)
    %     - num_epochs: number of epochs in training (default: 10)
    %     - classifier_parameters: a cell array of parameters to pass to
    %       the classifier's constructor

    %% Parse arguments
    parser = inputParser();
    parser.addParamValue('load_classifier', '', @ischar);
    parser.addParamValue('save_classifier', '', @ischar);
    parser.addParamValue('enable_training', true, @islogical);
    parser.addParamValue('enable_testing', true, @islogical);
    parser.addParamValue('num_epochs', 10, @isnumeric);
    parser.addParamValue('classifier_parameters', {}, @iscell);
    parser.parse(varargin{:});

    %% Initialize variables
    load_classifier = parser.Results.load_classifier;
    save_classifier = parser.Results.save_classifier;

    enable_training = parser.Results.enable_training;
    enable_testing = parser.Results.enable_testing;
    num_epochs = parser.Results.num_epochs;

    classifier_parameters = parser.Results.classifier_parameters;

    %% Load datasets
    if enable_training,
        try
            training_features = load_data_file( sprintf('%s-train.data', dataset_prefix) );
            training_labels = load_data_file( sprintf('%s-train.labels', dataset_prefix) );
        catch
            fprintf('Failed to load train set; disabling training!\n');
            training_features = [];
            training_labels = [];
            enable_training = false;
        end
    end

    if enable_testing,
        try
            testing_features = load_data_file( sprintf('%s-test.data', dataset_prefix) );
            testing_labels = load_data_file( sprintf('%s-test.labels', dataset_prefix) );
        catch
            fprintf('Failed to load test set; disabling testing!\n');
            testing_features = [];
            testing_labels = [];
            enable_testing = false;
        end
    end

    assert(enable_training || ~isempty(load_classifier), 'Neither training dataset nor pre-trained classifier provided!');

    %% Create classifier
    if ~isempty(load_classifier),
        % Load from file
        classifier = rofl.OrfSaffari.import_from_file(load_classifier);
    else
        % Create new
        num_classes = numel( unique(training_labels) );
        num_features = size(training_features, 1);
        classifier = rofl.OrfSaffari(num_classes, num_features, classifier_parameters{:});
    end

    %% Training
    if enable_training,
        t = tic();
        classifier.train(training_features, training_labels, ...
            'num_epochs', num_epochs, ...
            'verbose', true, ...
            'test_features', testing_features, ...
            'test_labels', testing_labels);
        t = toc(t);

        fprintf('Elapsed time: %f seconds\n', t);
    end

    %% Save classifier
    if ~isempty(save_classifier),
        classifier.export_to_file(save_classifier);
    end

    %% Test
    % Only if we didn't do the training...
    if ~enable_training && enable_testing,
        t = tic();
        predicted_labels = classifier.predict(testing_features);
        t = toc(t);

        incorrect = sum(predicted_labels ~= testing_labels);

        fprintf('Test error: %d/%d (%.02f %%)\n', incorrect, numel(testing_labels), incorrect/numel(testing_labels)*100);
        fprintf('Elapsed time: %f seconds\n', t);
    end
end

function data = load_data_file (filename)
    fid = fopen(filename, 'r');

    header = fgetl(fid);
    header_values = sscanf(header, '%d %d');

    num_samples = header_values(1);
    num_features = header_values(2);

    data = textscan(fid, '%f', inf);
    data = data{1};

    data = reshape(data, [ num_features, num_samples ]);
end
