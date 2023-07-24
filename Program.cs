using Microsoft.Extensions.ML;
using Microsoft.ML;
using SnowRoader.Models;
using SnowRoader.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddSingleton<MLContext>(sp => new MLContext());
builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
    .FromFile(filePath: Path.Combine(Directory.GetCurrentDirectory(), "workspace", "model.zip"), watchForChanges: true);

builder.Services.AddSingleton<IModelBuilderService, ModelBuilderService>();


var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
