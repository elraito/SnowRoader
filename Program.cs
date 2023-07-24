using Microsoft.Extensions.ML;
using Microsoft.ML;
using SnowRoader.Models;
using SnowRoader.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddSingleton<MLContext>(sp => new MLContext());

if (builder.Environment.IsProduction())
{
    builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
        .FromUri("https://snowderstorage.blob.core.windows.net/ml-model/dnn_model.zip");
}
else
{
    builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
        .FromFile("workspace/dnn_model.zip");
}

builder.Services.AddSingleton<IModelBuilderService, ModelBuilderService>();


var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
