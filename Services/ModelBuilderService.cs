
using System.Text.Json;
using Microsoft.Extensions.ML;
using Microsoft.ML;
using Microsoft.ML.Vision;
using SnowRoader.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace SnowRoader.Services;

public interface IModelBuilderService
{
    void TrainModel();
    ModelOutput Predict(byte[] image, string fileName);
}

public class ModelBuilderService : IModelBuilderService
{
    private readonly MLContext _mlContext;
    private readonly ILogger<ModelBuilderService> _logger;
    private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;

    public ModelBuilderService(MLContext mlContext, ILogger<ModelBuilderService> logger, PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool)
    {
        _mlContext = mlContext;
        _logger = logger;
        _predictionEnginePool = predictionEnginePool;
    }

    public void TrainModel()
    {
        var projectDirectory = Directory.GetCurrentDirectory();
        var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
        var assetsRelativePath = Path.Combine(projectDirectory, "dataset");

        IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

        IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);

        IDataView shuffledData = _mlContext.Data.ShuffleRows(imageData);

        var preprocessingPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelAsKey")
            .Append(_mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: assetsRelativePath,
                inputColumnName: "ImagePath"));

        IDataView preProcessedData = preprocessingPipeline
            .Fit(shuffledData)
            .Transform(shuffledData);

        TrainTestData trainSplit = _mlContext.Data.TrainTestSplit(preProcessedData, testFraction: 0.3);
        TrainTestData validationTestSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet);

        IDataView trainSet = trainSplit.TrainSet;
        IDataView validationSet = validationTestSplit.TrainSet;
        IDataView testSet = validationTestSplit.TestSet;

        var classifierOptions = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelAsKey",
            ValidationSet = validationSet,
            Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
            MetricsCallback = (metrics) => Console.WriteLine(metrics),
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true
        };

        var trainingPipeline = _mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        ITransformer trainedModel = trainingPipeline.Fit(trainSet);


        _mlContext.Model.Save(trainedModel, trainSet.Schema, Path.Combine(workspaceRelativePath, "model.zip"));

        // Use model

        ClassifySingleImage(_mlContext, testSet, trainedModel);
        ClassifyImages(_mlContext, testSet, trainedModel);
    }

    public ModelOutput Predict(byte[] image, string fileName)
    {

        var input = new ModelInput { Label = fileName, Image = image };

        var prediction = _predictionEnginePool.Predict(input);

        return prediction;
    }

    void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
    {
        PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
        ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
        ModelOutput prediction = predictionEngine.Predict(image);
        Console.WriteLine("Classifying single image");
        OutputPrediction(prediction);
    }

    void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
    {
        IDataView predictionData = trainedModel.Transform(data);
        IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);
        Console.WriteLine("Classifying multiple images");
        foreach (var prediction in predictions)
        {
            OutputPrediction(prediction);
        }
    }

    private static void OutputPrediction(ModelOutput prediction)
    {
        string imageName = Path.GetFileName(prediction.ImagePath);
        Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        Console.WriteLine($"--- Probability:{JsonSerializer.Serialize(prediction.Probability.ToList())}");
    }

    private IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
    {
        var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

        foreach (var file in files)
        {
            if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                continue;

            var label = Path.GetFileName(file);

            if (useFolderNameAsLabel)
                label = Directory.GetParent(file)!.Name;
            else
            {
                for (int index = 0; index < label.Length; index++)
                {
                    if (!char.IsLetter(label[index]))
                    {
                        label = label.Substring(0, index);
                        break;
                    }
                }
            }

            yield return new ImageData()
            {
                ImagePath = file,
                Label = label
            };
        }
    }
}