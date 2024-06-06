% Define the explainable AI model
classdef ExplainableAI < handle
    properties
        model
    end

    methods
        function obj = ExplainableAI()
            % Initialize the explainable AI model
            obj.model = fitensemble(X, Y, 'Method', 'Bag');
        end

        function [y, explanation] = predict(obj, X)
            % Make a prediction using the explainable AI model
            y = predict(obj.model, X);
            explanation = obj.explain(X, y);
        end

        function explanation = explain(obj, X, y)
            % Generate an explanation for the prediction
            explanation = [];
            for i = 1:size(X, 2)
                feature_importance = permutation_importance(obj.model, X, y, i);
                explanation = [explanation; feature_importance];
            end
        end
    end
end

% Example usage
X = rand(100, 10);
Y = rand(100, 1);
ei = ExplainableAI();
[y, explanation] = ei.predict(X);
fprintf("Prediction: %f\n", y);
fprintf("Explanation:\n");
disp(explanation);
