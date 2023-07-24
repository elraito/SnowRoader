using Microsoft.AspNetCore.Mvc;
using SnowRoader.Services;

namespace SnowRoader.Controllers;

[ApiController]
[Route("[controller]")]
public class ModelBuilderController : ControllerBase
{

    private readonly ILogger<ModelBuilderController> _logger;
    private readonly IModelBuilderService _modelBuilderService;

    public ModelBuilderController(ILogger<ModelBuilderController> logger, IModelBuilderService modelBuilderService)
    {
        _logger = logger;
        _modelBuilderService = modelBuilderService;
    }

    [HttpGet("HealthCheck")]
    public ActionResult<string> Get() => Ok("Ok");

    [HttpGet("TrainModel")]
    public ActionResult TrainModel()
    {
        _modelBuilderService.TrainModel();
        return Ok();
    }

    [HttpPost("Predict")]
    public async Task<ActionResult> PredictAsync(IFormFile file)
    {
        if (file == null || file.Length == 0)
            return BadRequest("No file received.");

        byte[] fileData;
        using (var memoryStream = new MemoryStream())
        {
            await file.CopyToAsync(memoryStream);
            fileData = memoryStream.ToArray();
        }

        var result = _modelBuilderService.Predict(fileData, file.FileName);
        return Ok(result);
    }
}
